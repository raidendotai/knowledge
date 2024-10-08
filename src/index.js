import axios from "axios";
import * as cheerio from "cheerio";
import PQueue from "p-queue";
import retry from "async-retry";
import crypto from "crypto";
import { URL } from "url";
import cliProgress from "cli-progress";
import path from "path";
import fs from "fs/promises";
import OpenAI from "openai";
import yaml from "yaml";
import dotenv from "dotenv";
import OpenAPIParser from "@readme/openapi-parser";
import slugify from "@sindresorhus/slugify";
import { PGlite } from "@electric-sql/pglite";
import { vector } from "@electric-sql/pglite/vector";

import pkg from "node-html-markdown";
const { NodeHtmlMarkdown, NodeHtmlMarkdownOptions } = pkg;
const nhm = new NodeHtmlMarkdown({});

dotenv.config();
import { get_encoding } from "tiktoken";
const enc = get_encoding("cl100k_base");

let openai;
try {
  openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
} catch (e) {
  console.error(e);
}

let LOCAL_PG_INSTANCE;
let LOCAL_KNOWLEDGE_DB = {};

const KNOWLEDGE_DIR = path.join(process.cwd(), ".knowledge");
const DB_DIR = path.join(KNOWLEDGE_DIR, "db");
const VECTORS_DIR = path.join(KNOWLEDGE_DIR, "vectors");
const INDEX_DIR = path.join(KNOWLEDGE_DIR, "index");

try {
  const files = await fs.readdir(VECTORS_DIR, { withFileTypes: true });
  const loadJsonFiles = async (dir) => {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await loadJsonFiles(fullPath);
      } else if (entry.isFile() && entry.name.endsWith('.json')) {
        const fileContent = await fs.readFile(fullPath, 'utf-8');
        const jsonData = JSON.parse(fileContent);
        const { meta, uid, content } = jsonData;
        LOCAL_KNOWLEDGE_DB[uid] = { meta, content };
      }
    }
  };

  await loadJsonFiles(VECTORS_DIR);
} catch (e) {
  console.error(`> no previous vectors db data to load`);
}

const initDirs = async () => {
  await fs.mkdir(DB_DIR, { recursive: true });
  await fs.mkdir(VECTORS_DIR, { recursive: true });
  await fs.mkdir(INDEX_DIR, { recursive: true });
};

await initDirs();

const queues = {
  search: new PQueue({ concurrency: 5 }),
  llm: new PQueue({ concurrency: 2 }),
  embed: new PQueue({ concurrency: 10 }),
};

const SERPER_API_KEY = process.env.SERPER_API_KEY;
if (!SERPER_API_KEY) {
  console.error("Error: SERPER_API_KEY is not set in .env file.");
  process.exit(1);
}

const serperInstance = axios.create({
  baseURL: "https://google.serper.dev",
  headers: {
    "X-API-KEY": SERPER_API_KEY,
    "Content-Type": "application/json",
  },
  timeout: 10000,
});

const PROXY_URL = process.env.PROXY_URL || null;
const getAxiosInstance = async () => {
  const axiosConfig = {
    timeout: 10000,
    headers: {
      "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " +
        "AppleWebKit/537.36 (KHTML, like Gecko) " +
        "Chrome/113.0.0.0 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.9",
    },
    validateStatus: (status) => status >= 200 && status < 400,
  };

  if (PROXY_URL) {
    axiosConfig.proxy = false; // Disable default proxy handling
    axiosConfig.httpsAgent = new (await import("https-proxy-agent")).default(
      PROXY_URL,
    );
    axiosConfig.httpAgent = new (await import("https-proxy-agent")).default(
      PROXY_URL,
    );
  }

  return axios.create(axiosConfig);
};
const axiosInstance = await getAxiosInstance();
const crawlQueue = new PQueue({ concurrency: 5 });

const _chunkify = (array, size) => {
  const chunks = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
};

const lib = {
  utils: {
    search: async ({ query }) => {
      return queues.search.add(() =>
        retry(
          async () => {
            const response = await serperInstance.post("/search", { q: query });
            return response.data;
          },
          {
            retries: 3,
            factor: 2,
            minTimeout: 1000,
            onRetry: (err, attempt) => {
              console.warn(
                `Search retry ${attempt} for query "${query}" due to ${err.message}`,
              );
            },
          },
        ),
      );
    },
    llm: async ({
      model = "gpt-4o-mini",
      messages,
      stream = process.stdout,
    }) => {
      return queues.llm.add(() =>
        retry(
          async () => {
            let opts = {
              model,
              messages,
            }
            if (!model.startsWith('o1')) {
              opts.stream = true
              opts.stream_options = { include_usage: true }
              const streaming = await openai.chat.completions.create(opts);

              let text = "";
              for await (const chunk of streaming) {
                const content = chunk.choices[0]?.delta?.content || "";
                if (content) {
                  text += content;
                  stream.write(content);
                }
              }
              stream.write(`\n`);
              return text.trim();
            } else {
              const response = await openai.chat.completions.create(opts);
              return response.choices[0]?.message?.content.trim() || "";
            }
          },
          {
            retries: 3,
            factor: 2,
            minTimeout: 1000,
            onRetry: (err, attempt) => {
              console.warn(
                `LLM retry ${attempt} due to ${err.message}`,
              );
            },
          },
        ),
      );
    },
    embed: async ({ texts, model = "text-embedding-3-small" }) => {
      const maxTokens = 8192; // Set maximum tokens limit

      const sliceTexts = (texts) => {
        return texts.map((text) => {
          const tokens = enc.encode(text); // Tokenize the text
          const txt = new TextDecoder().decode(
            enc.decode(tokens.slice(0, maxTokens)),
          );
          return txt; // Slice to max tokens and decode back to text
        });
      };

      texts = sliceTexts(texts);

      return queues.embed.add(() =>
        retry(
          async () => {
            const response = await openai.embeddings.create({
              model,
              input: texts,
              encoding_format: "float",
            });
            return {
              vectors: response.data
                .sort((a, b) => a.index - b.index)
                .map((e) => e.embedding),
              usage: { model, ...response.usage },
            };
          },
          {
            retries: 3,
            factor: 2,
            minTimeout: 1000,
            onRetry: (err, attempt) => {
              console.warn(`embed retry ${attempt} due to ${err.message}`);
            },
          },
        ),
      );
    },
    process: {
      html: async ({ url, proxy = false, use_puppeteer = false }) => {},
      typescript: async ({ name }) => {},
    },
  },
  collect: {
    crawl: async ({
      url,
      proxy = false,
      post_process = false,
      use_puppeteer = false,
      vectorize = false,
      index = false,
    }) => {
      const visited = new Set();
      const docs = [];
      const queue = new PQueue({ concurrency: 5 });

      const root_slug = url.split("://").slice(1).join("://").split("/")[0];

      // Progress bar setup
      const progressBar = new cliProgress.SingleBar({
        format: "Crawling |{bar}| {value}/{total} Pages",
        barCompleteChar: "\u2588",
        barIncompleteChar: "\u2591",
        hideCursor: true,
      });
      progressBar.start(1, 0);

      // URL normalization function
      const normalizeUrl = (inputUrl) => {
        try {
          const parsedUrl = new URL(inputUrl);
          parsedUrl.hash = ""; // Remove fragment
          parsedUrl.pathname = parsedUrl.pathname.replace(/\/$/, ""); // Remove trailing slash
          return parsedUrl.toString();
        } catch (e) {
          return null; // Invalid URL
        }
      };

      // Process a single URL
      const processUrl = async (currentUrl) => {
        const normalizedUrl = normalizeUrl(currentUrl);
        if (!normalizedUrl) {
          console.warn(`Invalid URL skipped: ${currentUrl}`);
          return;
        }

        try {
          const response = await retry(
            async () => {
              return await axiosInstance.get(normalizedUrl);
            },
            {
              retries: 2,
              factor: 2,
              minTimeout: 1000,
              onRetry: (err, attempt) => {
                console.warn(
                  `Retry ${attempt} for ${normalizedUrl} due to ${err.message}`,
                );
              },
            },
          );

          const html = response.data;
          const $ = cheerio.load(html);
          const meta = {
            title: $("title").text() || "No title available",
            description:
              $('meta[name="description"]').attr("content") ||
              "No description available",
            url: normalizedUrl,
          };
          const bodyHtml = $("body").html(); // Get the body content
          const markdown = nhm.translate(bodyHtml); // Convert body content to markdown
          console.log(` > ${normalizedUrl}`);
          docs.push({ url: normalizedUrl, content: markdown });
          if (markdown.length) {
            const slug = slugify(
              normalizedUrl.split("://").slice(1).join("://"),
            );
            const dir = path.join(DB_DIR, "crawl", root_slug, "raw");
            await fs.mkdir(dir, { recursive: true });
            await fs.writeFile(
              path.join(dir, `${slug}.yaml`),
              yaml.stringify({
                meta,
                content: markdown,
              }),
              "utf-8",
            );
          }

          // Find all internal links
          const baseUrl = new URL(url);
          $("a[href]").each((_, elem) => {
            const href = $(elem).attr("href") || "";
            if (
              href.startsWith("mailto:") ||
              href.startsWith("tel:") ||
              href.startsWith("javascript:")
            ) {
              return; // Skip non-HTTP links
            }
            try {
              const link = new URL(href, normalizedUrl);
              if (link.origin === baseUrl.origin) {
                // Only process same-origin links
                const normalizedLink = normalizeUrl(link.toString());
                if (normalizedLink && !visited.has(normalizedLink)) {
                  visited.add(normalizedLink);
                  queue.add(() => processUrl(normalizedLink));
                  progressBar.increment();
                  progressBar.setTotal(progressBar.getTotal() + 1);
                }
              }
            } catch (e) {
              // Ignore invalid URLs
            }
          });
        } catch (error) {
          console.error(`Failed to process ${normalizedUrl}: ${error.message}`);
        }
      };

      // Start crawling from the root URL
      const startingUrl = normalizeUrl(url);
      if (!startingUrl) {
        throw new Error("Invalid root URL provided.");
      }
      visited.add(startingUrl);
      queue.add(() => processUrl(startingUrl));

      await queue.onIdle();
      progressBar.stop();

      console.dir({ crawl: { url, done: true } });

      // Post-process if enabled
      if (post_process) {
        await lib.post_process({});
      }

      if (vectorize) {
        await lib.vectorize({ root: `crawl/${root_slug}` });

        if (index) {
          const index_methods = Object.keys(index).filter(key => index[key] !== false);
          await Promise.all(
            index_methods.map(async (index_method) => {
              await lib.index.create[index_method]({ root: `crawl/${root_slug}` });
            })
          )
        }
      }


      return;
    },
    learn: async ({
      query,
      proxy = false,
      post_process = false,
      use_puppeteer = false,
      vectorize = false,
      index = false,
    }) => {
      // Generate search queries using LLM
      const searchPrompt = `Generate a list of exactly ${process.env.SERPER_SEARCH_QUERIES} search queries to find information about:\n"${query}"\n\nin text format. For example:\n\`\`\`txt\n- example query- another query\n\`\`\`\n\n> do not wrap the search queries between quotes , should be raw text , one query per line\nyou are to write a total of : ${process.env.SERPER_SEARCH_QUERIES} search queries`;
      const searchQuerieResponse = await lib.utils.llm({
        messages: [{ role: "user", content: searchPrompt }],
      });
      console.dir({ searchQuerieResponse });

      const queries = searchQuerieResponse
        .split("\n")
        .map((l) => l.trim())
        .filter((line) => line.startsWith("-"))
        .map((line) => line.split("-").slice(1).join("-").trim())
        .filter((q) => q.length > 0)
        .slice(0,parseInt(process.env.SERPER_SEARCH_QUERIES));
      // console.dir({queries})

      // Perform searches
      const searchResults = [];
      for (const q of queries) {
        const result = await lib.utils.search({ query: q });
        searchResults.push(result);
      }

      // Collect HTML content from search results
      const docs = [];
      for (const res of searchResults) {
        if (res.organic) {
          // Assuming Serper returns organic results
          for (const item of res.organic) {
            docs.push({ url: item.link });
          }
        }
      }

      const root_slug = slugify(query);
      // Save and process the documents as needed
      for (const doc of docs) {
        try {
          const response = await retry(
            async () => {
              return await axiosInstance.get(doc.url);
            },
            {
              retries: 2,
              factor: 2,
              minTimeout: 1000,
              onRetry: (err, attempt) => {
                console.warn(
                  `Retry ${attempt} for ${doc.url} due to ${err.message}`,
                );
              },
            },
          );

          const html = response.data;
          const $ = cheerio.load(html);
          const meta = {
            title: $("title").text() || "No title available",
            description:
              $('meta[name="description"]').attr("content") ||
              "No description available",
            url: doc.url,
          };
          const bodyHtml = $("body").html(); // Get the body content
          const markdown = nhm.translate(bodyHtml); // Convert body content to markdown
          console.log(` > ${doc.url}`);
          if (markdown.length) {
            const slug = slugify(doc.url.split("://").slice(1).join("://"));
            const dir = path.join(DB_DIR, "learn", root_slug, "raw");
            await fs.mkdir(dir, { recursive: true });
            await fs.writeFile(
              path.join(dir, `${slug}.yaml`),
              yaml.stringify({
                meta,
                content: markdown,
              }),
              "utf-8",
            );
          }
        } catch (error) {
          console.error(`Failed to process ${doc.url}: ${error.message}`);
        }
      }

      console.dir({ learn: { query, done: true } });

      if (post_process) {
        await lib.post_process({});
      }

      if (vectorize) {
        await lib.vectorize({ root: `learn/${root_slug}` });
        if (index) {
          const index_methods = Object.keys(index).filter(key => index[key] !== false);
          await Promise.all(
            index_methods.map( async(index_method)=>{
              await lib.index.create[index_method]({ root: `learn/${root_slug}` });
            })
          )
        }
      }

      return;
    },
    openapi: async ({
      url,
      proxy = false,
      post_process = false,
      vectorize = false,
      index = false,
    }) => {
      const _circularReplacer = () => {
        const visited = new WeakSet();
        return (key, value) => {
          if (typeof value === "object" && value !== null) {
            if (visited.has(value)) {
              return;
            }
            visited.add(value);
          }
          return value;
        };
      };

      function openapi3(query) {
        const api = query.data;
        if (api.components) delete api.components;
        const _api = {
          openapi: api.openapi,
          servers: api.servers,
          info: api.info,
          tags: api.tags ? api.tags : false,
        };
        const paths = { ...api.paths };
        const api_functions = Object.keys(paths).map((path_key) => {
          return Object.keys(paths[path_key]).map((method_key) => {
            let descriptions = [];
            if (paths[path_key].summary)
              descriptions.push(paths[path_key].summary);
            if (paths[path_key].description)
              descriptions.push(paths[path_key].description);
            if (paths[path_key][method_key].summary)
              descriptions.push(paths[path_key][method_key].summary);
            if (paths[path_key][method_key].description)
              descriptions.push(paths[path_key][method_key].description);

            let category = false;
            try {
              category = api.info["x-apisguru-categories"][0];
            } catch (e) {
              false;
            }

            const openapi_specs = {
              ..._api,
              paths: {
                [path_key]: {
                  [method_key]: paths[path_key][method_key],
                },
              },
            };

            const _specs_string = JSON.stringify(openapi_specs).toLowerCase();

            const auth = [
              `Auth`,
              `Bearer`,
              `X-API-Key`,
              `X-Api-Key`,
              `BasicAuth`,
              `ApiKeyAuth`,
              `OpenID`,
              `OAuth2`,
            ].some((_auth_substring) =>
              _specs_string
                .toLowerCase()
                .includes(_auth_substring.toLowerCase()),
            );

            const _apiFunction = paths[path_key][method_key].operationId
              ? paths[path_key][method_key].operationId
              : `${path_key} : ${method_key}`;
            const _apiFunctionDescription = descriptions.join(`\n\n`);
            const _apiFunctionVectorize =
              `api info:\n` +
              `- name : ${api.info.title}\n` +
              `- description : ${
                api.info.description ? api.info.description.trim() : ""
              }\n` +
              `---\n` +
              `function name:\n` +
              `- ${_apiFunction.trim()}\n` +
              `---\n` +
              `function description:\n` +
              `- ${_apiFunctionDescription.trim()}\n` +
              `--\n` +
              `function route:\n` +
              `${Object.keys(
                openapi_specs.paths[Object.keys(openapi_specs.paths)[0]],
              )} : ${Object.keys(openapi_specs.paths)[0]}`;
            return {
              meta: {
                format: `openapi`,
                api: api.info.title,
                info: api.info.description ? api.info.description : ``,
                description: _apiFunctionVectorize,
                function: _apiFunction,
                urls: query.url ? [query.url] : [],
              },
              content: yaml.stringify({
                format: `openapi`,
                api: {
                  name: api.info.title,
                  description: api.info.description ? api.info.description : ``,
                },
                function: _apiFunction,
                description: _apiFunctionDescription,
                category,
                auth,
                implement: {
                  usage: [],
                  openapi: openapi_specs,
                },
              }),
            };
          });
        });
        return api_functions.flat();
      }

      async function run(query, attempt = 0) {
        if (attempt === 2) return false;

        if (!query.data) {
          if (!query.url) return false;
          query.data = (await axios.get(query.url)).data;
          query.data =
            typeof query.data === "string"
              ? yaml.parse(query.data)
              : query.data;
        }

        try {
          query.data = await OpenAPIParser.validate(query.data);
        } catch (e) {
          console.error(e, "fallback");
          query.data = await OpenAPIParser.parse(query.data);
        }
        query.data = JSON.parse(
          JSON.stringify(query.data, _circularReplacer()),
        );
        if (query.data.swagger) {
          if (!query.url) return false;
          attempt++;
          return await run(
            {
              url: `https://converter.swagger.io/api/convert?url=${query.url}`,
            },
            attempt,
          );
        }
        return openapi3(query);
      }

      let parsed_openapi;
      try {
        parsed_openapi = await run({ url });
      } catch (error) {
        console.error(`failed openapi parse : ${url}: ${error.message}`);
        return [];
      }

      const root_slug = slugify(url.split("://").slice(1).join("://"));
      for (let api_fn of parsed_openapi) {
        const { meta, content } = api_fn;
        const slug = slugify(api_fn.meta.function);
        const dir = path.join(DB_DIR, "openapi", root_slug, "raw");
        await fs.mkdir(dir, { recursive: true });
        await fs.writeFile(
          path.join(dir, `${slug}.yaml`),
          yaml.stringify({
            meta,
            content,
          }),
          "utf-8",
        );
      }

      if (post_process) {
        await lib.post_process({});
      }
      if (vectorize) {
        await lib.vectorize({ root: `openapi/${root_slug}` });
        if (index) {
          const index_methods = Object.keys(index).filter(key => index[key] !== false);
          await Promise.all(
            index_methods.map(async (index_method) => {
              await lib.index.create[index_method]({ root: `openapi/${root_slug}` });
            })
          )
        }
      }
    },
    github: async ({
      url,
      depth = 3,
      proxy = false,
      post_process = false,
      vectorize = false,
      index = false,
    }) => {
      const GITHUB_API_KEY = process.env.GITHUB_API_KEY;
      if (!GITHUB_API_KEY) {
        console.error("Error: GITHUB_API_KEY is not set in .env file.");
        return [];
      }

      const axiosInstance = axios.create({
        headers: {
          Authorization: `${GITHUB_API_KEY}`,
        },
      });

      // Helper function to fetch content with retry
      async function fetchContentWithRetry(fetchUrl) {
        return await retry(
          async (bail) => {
            try {
              const response = await axiosInstance.get(fetchUrl);
              return response.data;
            } catch (error) {
              if (error.response && error.response.status < 500) {
                bail(new Error(`Non-retryable error: ${error.message}`));
                return;
              }
              throw error;
            }
          },
          { retries: 5 },
        );
      }

      // Recursive function to traverse directories up to specified depth
      async function traverseRepo(path, currentDepth) {
        if (currentDepth > depth) return [];

        const apiUrl = `https://api.github.com/repos/${owner}/${repo}/contents/${path}`;
        let contents;
        try {
          contents = await fetchContentWithRetry(apiUrl);
        } catch (error) {
          console.error(
            `Failed to fetch contents of ${path}: ${error.message}`,
          );
          return [];
        }

        let readmes = [];

        for (const item of contents) {
          if (item.type === "file" && item.name.toLowerCase() === "readme.md") {
            const rawUrl = `https://raw.githubusercontent.com/${owner}/${repo}/${defaultBranch}/${item.path}`;
            try {
              const readmeContent = await fetchContentWithRetry(rawUrl);
              readmes.push({ git_path: item.path, content: readmeContent });
            } catch (error) {
              console.error(
                `Failed to fetch README at ${item.path}: ${error.message}`,
              );
            }
          } else if (item.type === "dir") {
            const subReadmes = await traverseRepo(item.path, currentDepth + 1);
            readmes = readmes.concat(subReadmes);
          }
        }

        return readmes;
      }

      // Parse the repo URL to extract owner and repo
      let owner, repo;
      try {
        const parsedUrl = new URL(url);
        const pathSegments = parsedUrl.pathname
          .split("/")
          .filter((seg) => seg.length);
        if (pathSegments.length < 2) {
          throw new Error("Invalid GitHub repository URL.");
        }
        owner = pathSegments[0];
        repo = pathSegments[1].replace(/\.git$/, "");
      } catch (error) {
        console.error(`Invalid URL provided: ${error.message}`);
        return [];
      }

      // Fetch repository metadata to get default branch
      let defaultBranch = "main";
      try {
        const repoUrl = `https://api.github.com/repos/${owner}/${repo}`;
        const repoData = await fetchContentWithRetry(repoUrl);
        defaultBranch = repoData.default_branch;
      } catch (error) {
        console.error(`Failed to fetch repository metadata: ${error.message}`);
        return [];
      }

      // Start traversing the repository
      const readmeList = await traverseRepo("", 1);

      const root_slug = `github.${owner}.${repo}`;
      for (let readme_file of readmeList) {
        const { git_path, content } = readme_file;
        if (content.length) {
          const slug = slugify(git_path);
          const dir = path.join(DB_DIR, "crawl", root_slug, "raw");
          await fs.mkdir(dir, { recursive: true });
          await fs.writeFile(
            path.join(dir, `${slug}.yaml`),
            yaml.stringify({
              meta: {
                owner,
                repo,
                path: git_path,
              },
              content,
            }),
            "utf-8",
          );
        }
      }

      if (post_process) {
        await lib.post_process({});
      }

      if (vectorize) {
        await lib.vectorize({ root: `crawl/${root_slug}` });
        if (index) {
          const index_methods = Object.keys(index).filter(key => index[key] !== false);
          await Promise.all(
            index_methods.map(async (index_method) => {
              await lib.index.create[index_method]({ root: `crawl/${root_slug}` });
            })
          )
        }
      }
    },
    npm: async ({
      name,
      proxy = false,
      post_process = false,
      vectorize = false,
      index = false,
    }) => {
      // Fetch package README using npm registry API
      // Fetch TypeScript definitions and process them
      try {
        const response = await axios.get(`https://registry.npmjs.org/${name}`);
        const latest = response.data["dist-tags"].latest;
        const git_url =
          response.data.versions[latest].homepage.split("#readme")[0];
        console.dir({ latest, git_url });
        if (git_url)
          return await lib.collect.github({
            url: git_url,
            proxy,
            post_process,
            vectorize,
            index,
          });
        return;
      } catch (error) {
        console.error(
          `Failed to fetch npm README for ${name}: ${error.message}`,
        );
        return [];
      }
    },
  },
  post_process: async ({}) => {
    // Iterate over each entry and generate cleaner Markdown using LLM
    const entries = await fs.readdir(DB_DIR, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory()) {
        const contentPath = path.join(DB_DIR, entry.name, "content.md");
        try {
          const content = await fs.readFile(contentPath, "utf-8");
          const cleanerPrompt = `Improve the following markdown:\n\n${content}`;
          const cleaner = await lib.utils.llm({ prompt: cleanerPrompt });
          await fs.writeFile(contentPath, cleaner, "utf-8");
        } catch (error) {
          console.error(
            `Failed to post-process ${entry.name}: ${error.message}`,
          );
        }
      }
    }
    return { status: "> postprocessing done" };
  },
  vectorize: async ({ root }) => {
    const processedDir = path.join(DB_DIR, root, "processed");
    const rawDir = path.join(DB_DIR, root, "raw");
    const dirToRead = (await fs.stat(processedDir).catch(() => false))
      ? processedDir
      : rawDir;
    const entries = await fs.readdir(dirToRead, { withFileTypes: true });
    const dataset = await Promise.all(
      entries.map(async (entry) => {
        const filepath = path.join(dirToRead, entry.name);
        const filecontent = await fs.readFile(filepath, "utf8");
        const data = yaml.parse(filecontent);

        const uid = crypto
          .createHash("sha512")
          .update(data.content)
          .digest("hex");
        return {
          scope: root,
          filepath,
          filename: entry.name,
          uid,
          ...data,
          vector_text: `${yaml.stringify(data.meta)}\n---\n\n${data.content.trim()}`,
        };
      }),
    );
    const batches = _chunkify(dataset, 15);
    await Promise.all(
      batches.map(async (chunk, chunk_index) => {
        // console.dir({chunk , chunk_index})
        const vectors = (
          await lib.utils.embed({
            texts: chunk.map((entry) => entry.vector_text),
          })
        ).vectors;
        await Promise.all(
          vectors.map(async (vector, idx) => {
            const item = chunk[idx];
            const vectorDir = path.join(VECTORS_DIR, root);
            await fs.mkdir(vectorDir, { recursive: true });
            await fs.writeFile(
              path.join(VECTORS_DIR, root, `${item.uid}.json`),
              JSON.stringify({
                ...item,
                vector,
              }),
              "utf-8",
            );
            if (LOCAL_KNOWLEDGE_DB) {
              LOCAL_KNOWLEDGE_DB[item.uid] = { meta: item.meta , content: item.content }
            }
          }),
        );
      }),
    );
    console.log(`> vectorized : ${root}`);
  },
  index: {
    create: {
      local: async ({ root }) => {
        if (!LOCAL_PG_INSTANCE) {
          const metaDb = new PGlite(INDEX_DIR, {
            extensions: {
              vector,
            },
          });
          await metaDb.waitReady;
          LOCAL_PG_INSTANCE = metaDb;
        }
        try {
          await LOCAL_PG_INSTANCE.exec(`
            create extension if not exists vector;
            -- drop table if exists embeddings; -- Uncomment this line to reset the database
            create table if not exists embeddings (
              id bigint primary key generated always as identity,
              uid text not null unique,
              embedding vector (1536)
            );
            
            create index on embeddings using hnsw (embedding vector_ip_ops);
          `);
        } catch (e) {
          console.error(e);
        }
        const entries = await fs.readdir(path.join(VECTORS_DIR, root));
        const jsonFiles = entries.filter((file) => file.endsWith(".json"));
        const dataset = await Promise.all(
          jsonFiles.map(async (file) => {
            const filePath = path.join(VECTORS_DIR, root, file);
            const content = await fs.readFile(filePath, "utf-8");
            return JSON.parse(content);
          }),
        );
        const chunks = _chunkify(dataset, 50);
        for (let chunk of chunks) {
          // Filter out entries that already exist in the database
          const existingUids = await LOCAL_PG_INSTANCE.query(`
    SELECT uid FROM embeddings WHERE uid IN (${chunk.map((entry) => `'${entry.uid}'`).join(", ")});
  `);
          const existingUidSet = new Set(
            existingUids.rows.map((row) => row.uid),
          );

          const newEntries = chunk.filter(
            (entry) => !existingUidSet.has(entry.uid),
          );
          if (newEntries.length > 0) {
            const pg_entries = newEntries
              .map((entry) => {
                return `\t('${entry.uid}','${JSON.stringify(entry.vector)}')`;
              })
              .join(",\n");

            await LOCAL_PG_INSTANCE.exec(`
insert into embeddings (uid, embedding) values
  ${pg_entries};
`);
          }

          console.dir(
            await LOCAL_PG_INSTANCE.query(`SELECT COUNT(*) FROM embeddings;`),
            { depth: null },
          );
        }
      },
      supabase: async ({}) => {},
      weaviate: async ({}) => {},
    },
    query: {
      local: async ({ query , embedding = false, threshold = 0.0 , amount = 6  }) => {
        if (!LOCAL_PG_INSTANCE) {
          const metaDb = new PGlite(INDEX_DIR, {
            extensions: {
              vector,
            },
          });
          await metaDb.waitReady;
          LOCAL_PG_INSTANCE = metaDb;
        }
        const query_vector = embedding ? embedding : ( await lib.utils.embed({texts:[query]}) ).vectors[0]
        const res = await LOCAL_PG_INSTANCE.query(
          `
          select * from embeddings
          where embeddings.embedding <#> $1 < $2
          order by embeddings.embedding <#> $1
          limit $3;
          `,
          [JSON.stringify(query_vector), -Number(threshold), Number(amount)]
        )
        return res.rows.map(item=>{
          return {
            uid: item.uid,
            data: LOCAL_KNOWLEDGE_DB[item.uid] ? LOCAL_KNOWLEDGE_DB[item.uid] : false,
          }
        })
      },
      supabase: async ({}) => {},
      weaviate: async ({}) => {},
    },
    ask: {
      local: async({query , model="gpt-4o"}) => {

        const retrieved = await lib.index.query.local({
          query,
          amount: 5
        })
        const messages = [
          {
            role:'user',
            content: `# FOUND REFERENCES :

${retrieved.map(entry=>yaml.stringify(entry.data)).join('\n---\n')}
------

# USER QUERY :

${query}`
          }
        ]

        return await lib.utils.llm({messages , model});
      }
    },
  },
};

export default lib;
