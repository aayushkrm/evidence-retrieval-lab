# Evidence Retrieval Lab

An offline workbench for measuring document retrieval and searching your own text. It combines a versioned local document store, a passage-search API and dashboard, and experiments on the labeled [BEIR SciFact benchmark](https://github.com/beir-cellar/beir). The focus is **finding relevant documents**, not generating or verifying answers.

**Measured outcome:** On 300 SciFact test queries, hybrid search obtained Recall@10 of 0.814 versus 0.788 for BM25. A separate exploratory passage experiment did not show a clear gain over full-document BM25 and was slower on this machine. These results concern document retrieval on one previously inspected benchmark, not answer accuracy or production performance.

```text
SciFact corpus ──> BM25 ──────────────┐
                └> MiniLM embeddings ├──> ranked document IDs
                     └──> rank fusion ┘          │
                                         relevance judgments
                                                  ↓
                                  Recall / MRR / nDCG + paired analysis

Your text ──> SQLite collection ──> versioned snapshot ──> passage BM25 ──> API / dashboard
```

## Search your own documents

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[service,test]'
python -m evidence_lab.service
```

Open [localhost:8766](http://127.0.0.1:8766/). Create or select a collection, paste a JSON array of documents, and search. Each match includes the parent document ID and character offsets into its original text. The interactive API reference is at [localhost:8766/docs](http://127.0.0.1:8766/docs).

The service stores documents in `data/workspace.sqlite3` by default; `EVIDENCE_LAB_DB` selects another local path. Updates and deletes are atomic. An optional `expected_generation` rejects stale writes with HTTP 409. A monotonically increasing generation identifies the snapshot used by each search; it is not a historical backup. The audit log records operation counts and times, not document text. The in-memory search index is rebuilt after a collection changes.

```bash
curl -s http://127.0.0.1:8766/v1/tenants/local/collections/notes/documents \
  -H 'Content-Type: application/json' \
  -d '{"documents":[{"id":"intro","title":"Example","text":"Solar panels convert sunlight to electricity."}]}'
curl -s http://127.0.0.1:8766/v1/tenants/local/collections/notes/search \
  -H 'Content-Type: application/json' -d '{"query":"sunlight"}'
```

The service is a **single-user local prototype**, not a secure multi-user deployment. Tenant names are namespaces, not authentication: any local caller can choose one. SQLite content is unencrypted. The default server binds to `127.0.0.1`; it checks Host and browser Origin, but has no user authentication, authorization, TLS, or production rate limiting. Do not expose it to a network or store sensitive documents without adding and reviewing those controls. Search accepts collections up to 2 MB of combined title and text; storage supports larger collections. Scores can include a title match, so an excerpt is not necessarily evidence for a claim.

## Run the local demo

Requires Python 3.10+. The default BM25 path has no third-party runtime dependencies. The first command downloads data from the BEIR host, checks the publisher's archive checksum, and writes to a new local directory. Review the dataset license notes below before use.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
evidence-lab download
python -m evidence_lab.web
```

Open [localhost:8765](http://127.0.0.1:8765/). The browser demo performs BM25 search only. It binds to loopback, returns matching abstracts, and does not call an LLM. Search results are not verified claims and must not be used as medical advice.

The command-line interface also supports `evidence-lab search "your query" --method bm25`. Install `pip install -e '.[dense]'` to use `--method dense` or `--method hybrid`. These methods download an approximately 90 MB pretrained model and build embeddings on CPU; first use can take several minutes. The default model is `sentence-transformers/all-MiniLM-L6-v2`, pinned to a model revision in code. Dense retrieval truncates long documents at the model's maximum sequence length. See the model's [card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for its intended use and limitations.

For repeated dense searches or experiments, pass `--cache-dir cache` to the search or evaluation command. The opt-in cache saves corpus embeddings, keyed by the corpus, immutable 40-character model revision and relevant software versions. A later run with the same configuration can reuse them, although it still loads the model to encode queries. Keep the cache directory trusted: non-pickle loading and structural checks do not authenticate its contents or make hostile files safe. If a cache is reported invalid, remove the identified cache file and rebuild it from the trusted corpus and model.

## Reproduce an experiment

```bash
pip install -e '.[service,test]'
python -m scripts.evaluate --data data/scifact --split dev --systems bm25 --output results/dev-bm25.json
# Optional: install the dense extra first.
python -m scripts.evaluate --data data/scifact --split dev --systems bm25 dense hybrid --output results/dev-all.json
python -m unittest discover -s tests -v
```

The development set is a deterministic 20% sample of the official training query IDs. For final reporting only, after choosing methods and settings on train/dev, use `--split test --final-test --overlap-sensitivity`. The switch records the user's intent; it cannot prevent someone from inspecting test labels or retuning after seeing results. Reports contain query-level metrics, document IDs, timing, dataset hashes, software versions, and paired bootstrap intervals. Local reports may include dataset text and are not committed.

The benchmark's original train and test splits contain **two query pairs with the same text after case and whitespace normalization**, and the paired queries have identical relevance judgments. The sensitivity option reports both the full official test and results excluding those test queries. It cannot detect all near-duplicates or establish that a pretrained model has never encountered benchmark content.

For each judged query, the evaluator measures Recall@k, reciprocal rank, and nDCG@k; it treats unjudged documents as non-relevant and excludes queries without positive labels. It uses exponential gain `2^grade−1`; SciFact's labels are binary. Latency is sequential warmed query time on one machine, excluding model download and index construction. A paired query bootstrap estimates uncertainty in score differences; it is not proof that a system will work in another domain.

## Current result and scope

On the official 300-query test set, the fixed systems produced:

| Method | Recall@10 | MRR@10 | nDCG@10 | Median query time |
|---|---:|---:|---:|---:|
| BM25 | 0.788 | 0.631 | 0.663 | 13.3 ms |
| MiniLM dense | 0.783 | 0.605 | 0.645 | 16.6 ms |
| Hybrid rank fusion | 0.814 | 0.646 | 0.683 | 32.0 ms |

The paired 95% query-bootstrap interval for hybrid minus BM25 Recall@10 was **+0.002 to +0.052**. For nDCG@10, the interval includes zero. These exploratory intervals do not account for multiple comparisons, dataset selection, or possible pretraining exposure; they are not proof of general improvement. Excluding the two normalized-text overlaps gave Recall@10 of 0.790 / 0.785 / 0.816 for BM25 / dense / hybrid. Query times exclude model loading and indexing. These are retrieval scores, not answer-quality or claim-verification scores. See [the experiment card](docs/experiment.md) for development results and the full setup.

## Data and licensing

The source dataset is SciFact ([original project](https://github.com/allenai/scifact)), distributed here through the [BEIR benchmark](https://github.com/beir-cellar/beir). This repository downloads the dataset rather than redistributing it. The original [SciFact license](https://github.com/allenai/scifact/blob/master/LICENSE.md) distinguishes claims and annotations from abstracts; the [AllenAI dataset card](https://huggingface.co/datasets/allenai/scifact) lists a different license. Review applicable rights before redistribution or commercial use; downloading does not resolve that discrepancy.

The [industry research](docs/industry-research.md) explains the project selection, current hiring evidence, alternative directions and limits. This remains a research prototype; it has no authentication, document permissions, automatic evaluation, or production deployment.
