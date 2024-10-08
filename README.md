## Overview

- `@openinterface/knowledge` npm package repo
- say hi 👋 [@n_raidenai](https://x.com/n_raidenai)

## knowledge

- agent tool to autonomously learn how to use APIs, SDKs, infra tools , ...
- collects documentation for RAG, as it enables
  - crawling docs websites
  - crawling github repos for readmes, npm from package name
  - searching for use cases (via serper) from a single query
  - parse openapi/swagger definitions from urls
- automatically manages vectorizing , embedding , indexing , concurrency
- has local index powered by `@electric-sql/pglite` and `pgvector`
- (wip) post processes collected documents to clean up and improve formatting
- (wip) stores in remote index dbs (like supabase , weaviate , ... ) 

## Installation

```bash
npm install @openinterface/knowledge
```

## Usage

make a .env file, ensure it has these values

```env
OPENAI_API_KEY = "REPLACE_KEY" # required

SERPER_API_KEY = "REPLACE_KEY" # to enable knowledge.collect.learn feature
SERPER_SEARCH_QUERIES = 2 # search queries per learn operation (if enabled)

GITHUB_API_KEY = "REPLACE_KEY" # to enable knowledge.collect.github feature

#PROXY_URL = http://your_proxy_url:port # optional , for scraping / crawling pages
```

import as follows

```javascript
import knowledge from '@openinterface/knowledge';
```

## Use Case Examples

```javascript
import knowledge from '@openinterface/knowledge';

// ====================================================================
// FETCHING DOCS / DATA + EMBEDDING RESULTS + INDEXING IN VECTOR DB
// ====================================================================

// collect + index documentation by crawling a website's docs
await knowledge.collect.crawl({
  url: 'https://docs.railway.app/',
  vectorize: true,
  index: {
    local: true,
    // postgres : false, // remote index not implemented yet
    // weaviate : false, // remote index not implemented yet
  },
});

// collect + index tutorials/articles/docs by googling a use case (needs serper key in .env)
await knowledge.collect.learn({
  query: 'setup and deploy graphql with node',
  vectorize: true,
  index: { local: true },
});;

// collect + index readmes from a github (needs github key in .env)
await knowledge.collect.github({
  url: 'https://github.com/resend/react-email',
  vectorize: true,
  index: { local: true },
});
// collect + index readmes from a npm , by crawling its assigned github repo (needs github key in .env)
await knowledge.collect.npm({
  name: 'react-confetti',
  vectorize: true,
  index: { local: true },
});

// collect + index every {method,route} combination from an openapi specifications file url (can be yaml or json)
await knowledge.collect.openapi({
  url: 'https://raw.githubusercontent.com/resend/resend-openapi/refs/heads/main/resend.yaml',
  vectorize: true,
  index: { local: true },
});


// ====================================================================
// QUERYING THE COLLECTED DATA
// ====================================================================

// search example
const retrieved = await knowledge.index.query.local({
  query: "create graphql schemas for invoices",
  amount: 4
})
/*
  -> retrieved : 
  [
    {
      uid,
      data: {
        meta: {...}
        content: "... documentation content ..."
      },
    },
    ...
  ]
*/

// RAG example
const answer = await knowledge.index.ask.local({
  query: `make a new nodejs project that :

> makes a local vectra index
> indexes from a csv list of my clients , which is 'name,email,phone,task_description'
> write test cases ;

no typescript, and use type : module

answer with the new , entire project codebase , with every file needed (including any example), in format :
\`\`\`yaml
repo:
  - path: "" # full file path
    content: "" # full file content
  - ...
\`\`\``,
  model: `o1-mini`
})
console.dir({answer})
```

## Potential Issues

- if using the local index features (and that depend on `@electric-sql/pglite` and `@electric-sql/pglite/pgvector`) in a cloud dockerized environment, might run into some issues.
  the npm installer for pgvector does not handle the full installation by default
- although, should work without problem in local / browsers envs

## WIP

- post processing retrieved documents (clean up and reformat with LLM)
- indexing in remote vector database (supabase , weaviate)