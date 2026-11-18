# Evidence Retrieval Lab

An offline workbench for measuring document retrieval. It compares keyword, neural and hybrid retrieval on the labeled [BEIR SciFact benchmark](https://github.com/beir-cellar/beir), and includes a small local search demo. The focus is **finding relevant documents**, not generating or verifying answers.

```text
SciFact corpus ──> BM25 ──────────────┐
                └> MiniLM embeddings ├──> ranked document IDs
                     └──> rank fusion ┘          │
                                         relevance judgments
                                                  ↓
                                  Recall / MRR / nDCG + paired analysis
```

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
python -m scripts.evaluate --data data/scifact --split dev --systems bm25 --output results/dev-bm25.json
# Optional: install the dense extra first.
python -m scripts.evaluate --data data/scifact --split dev --systems bm25 dense hybrid --output results/dev-all.json
python -m unittest discover -s tests -v
```

The development set is a deterministic 20% sample of the official training query IDs. For final reporting only, after choosing methods and settings on train/dev, use `--split test --final-test --overlap-sensitivity`. The switch records the user's intent; it cannot prevent someone from inspecting test labels or retuning after seeing results. Reports contain query-level metrics, document IDs, timing, dataset hashes, software versions, and paired bootstrap intervals. Local reports may include dataset text and are not committed.

The benchmark's original train and test splits contain **two query pairs with the same text after case and whitespace normalization**, and the paired queries have identical relevance judgments. The sensitivity option reports both the full official test and results excluding those test queries. It cannot detect all near-duplicates or establish that a pretrained model has never encountered benchmark content.

For each judged query, the evaluator measures Recall@k, reciprocal rank, and nDCG@k; it treats unjudged documents as non-relevant and excludes queries without positive labels. It uses exponential gain `2^grade−1`; SciFact's labels are binary. Latency is sequential warmed query time on one machine, excluding model download and index construction. A paired query bootstrap estimates uncertainty in score differences; it is not proof that a system will work in another domain.

## Current result and scope

On the official 300-query test set, the fixed BM25 baseline achieved **Recall@10 0.788, MRR@10 0.631, and nDCG@10 0.663**. Excluding two queries with normalized-text matches in training gave Recall@10 0.790. These are retrieval scores, not answer-quality or claim-verification scores. See [the experiment card](docs/experiment.md) for the exact setup and limitations; add other systems only after measuring them.

On the 162-query development subset, the measured comparison was:

| Method | Recall@10 | nDCG@10 | Median query time |
|---|---:|---:|---:|
| BM25 | 0.824 | 0.711 | 14.6 ms |
| MiniLM dense | 0.805 | 0.694 | 16.7 ms |
| Hybrid rank fusion | 0.836 | 0.722 | 33.6 ms |

The paired 95% bootstrap interval for hybrid minus BM25 Recall@10 was **−0.031 to +0.056**. This small development sample does not establish that hybrid is better, and the time figures exclude indexing. Dense and hybrid were not evaluated on the official test set in this run.

## Data and licensing

The source dataset is SciFact ([original project](https://github.com/allenai/scifact)), distributed here through the [BEIR benchmark](https://github.com/beir-cellar/beir). This repository downloads the dataset rather than redistributing it. The original [SciFact license](https://github.com/allenai/scifact/blob/master/LICENSE.md) distinguishes claims and annotations from abstracts; the [AllenAI dataset card](https://huggingface.co/datasets/allenai/scifact) lists a different license. Review applicable rights before redistribution or commercial use; downloading does not resolve that discrepancy.

The [industry research](docs/industry-research.md) explains the project selection, current hiring evidence, alternative directions and limits. This is a research prototype, not a production service: it has no authentication, document permissions, tenant isolation, automatic evaluation, or live-corpus update pipeline.
