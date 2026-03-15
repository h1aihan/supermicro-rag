# Feature Update: eStore FAQ Ingestion for RAG Chatbot

## Problem

The current RAG chatbot has no coverage for eStore operational questions such as ordering, shipping, returns, payments, account management, or software licensing. These topics are covered in the Supermicro eStore FAQ page (`/us_en/faq/`), which contains 133 Q&A pairs across 10 categories.

However, the FAQ content is rendered through a JavaScript-driven accordion UI (Knockout.js). A simple HTTP GET only retrieves the page shell, not the full FAQ dataset, making it inaccessible to the crawler used by the RAG pipeline.

## Discovery

Inspection of the FAQ page revealed that selecting a category triggers a POST request to an internal AJAX endpoint:

```
POST /us_en/faq/category/getfaq/
Body: category_id=<N>
```

The endpoint returns a JSON payload of FAQ entries containing fields such as `faq_id`, `question`, `answer` (HTML), `category_id`, and `sort_order`.

Additionally, the page embeds the category list and an initial set of FAQ items inside a `<script>` block as part of the Knockout.js view-model. This means the full dataset can be retrieved using one page fetch plus one POST request per category, avoiding the need for a headless browser.

## System Design

The feature introduces an end-to-end pipeline that integrates FAQ content into the existing RAG architecture.

Pipeline stages: Crawling → Ingestion → Indexing → Query Planning → Retrieval → Answer Generation

### 1. Crawling (`scrape_estore_faq.py`)

A new crawler replicates the site's data-loading behavior.

The crawler reuses the same cloudscraper session and anti-bot patterns used by the accessories crawler. It fetches the FAQ page, extracts the category metadata and AJAX endpoint URL from the embedded Knockout.js JSON, and then sends POST requests for each of the 10 categories to retrieve FAQ entries.

If the AJAX request fails, the script falls back to the embedded FAQ items present in the page's script block.

Each FAQ entry is normalized into a structured JSON record containing `faq_id`, `category_name`, `question`, `answer_text`, and `source_url`. HTML answers are converted into clean plain text using BeautifulSoup so they can be embedded effectively for retrieval.

### 2. Ingestion (`scripts/ingest_faq.py`)

A companion ingestion script converts the scraped FAQ data into the existing RAG document schema (`rag_content.jsonl`).

Each FAQ pair becomes one RAG document. The FAQ question is used as both the document title and heading, while the answer becomes the content body. This ensures the question acts as strong retrieval keywords for BM25 search.

Each entry is tagged with `"FAQ - {category}"` so the system can distinguish FAQ documents from product documentation during retrieval.

The ingestion script also deduplicates entries by title before appending them to the corpus.

### 3. Indexing (`setup_rag.py`)

After ingestion, the existing indexing pipeline processes the FAQ documents alongside all other RAG content.

The pipeline chunks the documents, generates embeddings, builds the FAISS vector index, and updates the BM25 keyword index.

Since FAQ entries are short Q&A pairs, each record typically produces one or two chunks, making them naturally suited for retrieval.

### 4. Query Planning (`query_planner.py`)

A new intent classification type (`intent="faq"`) was added to the query planner.

When a user asks operational questions related to shipping, returns, ordering, payments, licensing, or accounts, the planner classifies the query as FAQ intent and routes it to the RAG system. In this case the planner sets:
- `use_catalog = false`
- `use_rag = true`

This prevents the system from loading product catalog data that is irrelevant to policy or operational questions.

A keyword-based fallback regex is also included to catch FAQ queries if the LLM planner fails to classify them correctly.

### 5. Retrieval (`chatbot.py` — `_retrieve_context()`)

Retrieval behavior is tuned for FAQ queries. When `intent="faq"`:
- `top_k` is reduced to 5
- Product catalog retrieval is disabled
- Entity graph expansion is disabled

Retrieval uses a two-pass combined approach:

**Pass 1 — FAQ Question Bank (title-level semantic matching)**

At index load time, FAQ question titles are extracted from chunk metadata, deduplicated, and embedded as sentence vectors using the same sentence-transformer model (`all-MiniLM-L6-v2`). These embeddings are stored in a lightweight in-memory matrix alongside a mapping from question index to chunk indices.

At query time, the user's question is encoded into a 384-dim sentence vector. Cosine similarity is computed against all FAQ title vectors via a single matrix dot product. The top-k FAQ titles by similarity score are returned, along with their associated content chunks.

This approach compares the user's question against short, clean FAQ titles rather than long body text, which allows the sentence-transformer to recognize semantic equivalence even when keywords differ entirely (e.g., "ship outside the US" matches "ship internationally").

**Pass 2 — Source-filtered hybrid search (body-level keyword + semantic matching)**

The standard 3-channel hybrid retrieval pipeline (FAISS semantic + BM25 keyword + filename index) runs with `source_filter="FAQ:"`, restricting all channels to FAQ chunks only. This searches the full body text of FAQ entries, catching indirect queries where keywords appear in the answer body but not the title (e.g., a query about "refund" matches a FAQ body that mentions "refund" 8 times, even if the title says "return policy").

**Merge**

Results from Pass 1 (question bank) take priority as the highest-confidence matches. Pass 2 (hybrid search) results backfill remaining slots after deduplication. This gives the system both precision (from title matching) and recall (from body keyword matching).

A small number of general-corpus supplement chunks (up to 2) are also appended for broader context when available.

### 6. Answer Generation (chatbot prompts)

Two prompt-level changes steer the LLM's output style.

The system prompt now includes a dedicated FAQ instruction section directing the LLM to respond in a concise customer-service tone and avoid unnecessary formatting such as markdown tables.

When FAQ intent is detected, the chatbot switches to an FAQ-specific prompt template that instructs the model to answer directly using the retrieved FAQ content instead of generating product specification style responses.

### 7. Testing

A suite of 16 FAQ test cases was added to `tests/test_product_queries.py` under `category="faq"`:

**8 direct FAQ tests** — straightforward questions that closely match FAQ topics:

| Test ID | Query | Expected Content |
|---------|-------|-----------------|
| `faq_return_policy` | What is the return policy for Supermicro eStore? | 30-day return window; software/lifestyle non-returnable; restocking fee up to 15% |
| `faq_shipping_international` | Does Supermicro ship internationally? | US and Canada only; does not ship to US territories |
| `faq_payment_methods` | What payment methods does Supermicro eStore accept? | Visa, MasterCard, American Express, Discover |
| `faq_cancel_order` | How do I cancel my order on the eStore? | Cancel via My Account > My Orders |
| `faq_warranty_servers` | What is the warranty on Supermicro eStore servers? | 3-year labor and parts; 1-year cross-shipping |
| `faq_software_license_key` | Where do I find and generate my software license key? | My Account > My Software |
| `faq_tax_exemption` | How do I apply for tax exemption on the eStore? | Apply through eStore account; reviewed within 1 business day |
| `faq_free_shipping` | Does Supermicro offer free shipping? | Free shipping within Continental US on purchases over $200 |

**8 wording-variation tests** — intentionally rephrase the same topics using different words to stress-test retrieval robustness:

| Test ID | Query | Target FAQ Topic |
|---------|-------|-----------------|
| `faq_var_refund_wording` | Can I get a refund if I changed my mind about a purchase? | Return/refund policy |
| `faq_var_guest_checkout` | Can I order from the eStore without creating an account? | Account requirements |
| `faq_var_backorder_partial_ship` | If part of my order is backordered, will you send what's available now? | Partial shipment policy |
| `faq_var_password_requirements` | My new password keeps getting rejected, what are the rules? | Password requirements |
| `faq_var_combine_accounts` | I have two eStore accounts, can you merge them? | Account merging |
| `faq_var_credit_card_types` | Can I pay with PayPal or a wire transfer? | Payment methods |
| `faq_var_track_my_order` | How can I check where my package is right now? | Order tracking |
| `faq_var_tax_canada_reseller` | I'm reselling from Canada, do I still get charged sales tax? | Canadian tax policy |

Tests are run via:

```bash
python tests/test_product_queries.py --category faq --summary
```

Each test verifies that the chatbot's response contains the expected factual content. The `--summary` flag provides quality ratings (Excellent/Good/Weak) based on keyword coverage of the expected answer.

### 8. Iterative Improvement

The retrieval strategy evolved through three iterations, driven by test results.

**Iteration 1 — Source-filtered hybrid search only**

The initial implementation used only Pass 2 (source-filtered hybrid search). For most tests this worked well, but 2 queries failed:
- `faq_shipping_international` — rated "Weak"
- `faq_software_license_key` — rated "Weak"

Root cause analysis revealed that the LLM query planner rewrites user queries before retrieval. "Does Supermicro ship internationally?" was rewritten to something like "FAQ international shipping policy" — which shares almost no keywords with the actual FAQ title "Does Supermicro eStore ship internationally?". BM25 found no term overlap, and FAISS semantic search on long chunk bodies diluted the signal.

**Iteration 2 — FAQ Question Bank only**

We built a dedicated FAQ question bank: a mini-index of just FAQ titles embedded as sentence vectors. At query time, the user's raw question (before planner rewriting) is compared directly against these title embeddings via cosine similarity. The sentence-transformer model, trained on semantic textual similarity benchmarks, recognizes that "ship outside the US" and "ship internationally" carry the same meaning even without shared keywords.

Result: The 2 failing tests improved to "Excellent." However, running the full 16-test suite revealed a regression — 3 tests dropped from "Excellent" to "Good":
- `faq_return_policy`
- `faq_var_refund_wording`
- `faq_var_track_my_order`

Root cause: Pure title matching lost the keyword signal that BM25 provided on body text. For example, a query about "getting my money back" has moderate semantic similarity to the title "What is the return and refund policy?" — but the FAQ body contains the word "refund" 8 times. BM25 on body text would have caught this easily; title-only cosine similarity missed it.

**Iteration 3 — Combined approach (shipped)**

Merging both passes resolved the tension between precision and recall:
- Pass 1 (question bank) provides high-confidence matches when the user's phrasing is semantically close to a FAQ title
- Pass 2 (hybrid search) catches indirect queries where keywords appear in the body but not the title
- Question bank results take priority; hybrid results backfill after deduplication

Result: The 2 original failures were fixed, and the 3 regressions recovered. The few remaining "Good" (vs "Excellent") ratings were confirmed to be LLM generation wording choices, not retrieval failures — the relevant FAQ content was present in the context.
