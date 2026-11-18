# Industry research and project selection

Research date: 19 November 2026. This is a targeted review of selected reports, job descriptions, and primary technical sources. It is not a representative survey of all vacancies, and a project cannot guarantee interviews or offers. Listings may close and have location, education, and work-authorization requirements.

## What employers ask for

- [AppLovin's 2027 ML engineering internship](https://job-boards.greenhouse.io/applovin/jobs/4713038006) describes end-to-end ML work, model evaluation, ranking and experimentation, Python, and software engineering fundamentals. It accepts bachelor's or master's students in relevant fields.
- [NVIDIA's 2027 AI agents and backend infrastructure internship](https://jobs.nvidia.com/careers/job/893398376165) describes backend services, agent reliability and evaluation, API design, databases, and software fundamentals. Its listing targets bachelor's or master's students.
- [Toyota Research Institute's multi-agent research internship](https://jobs.lever.co/tri/86e6a285-bb0a-4e84-8bf9-bd498d45c33d) calls for controlled comparisons and measurements such as accuracy and compute; it is a doctoral role, so it illustrates technical direction rather than a broadly eligible internship.
- The [2026 Stanford AI Index economy chapter](https://hai.stanford.edu/assets/files/ai_index_report_2026_chapter_4_economy.pdf) analyzes AI-related job postings. Such postings are not hires, the report's US figures do not represent every country, and skills can overlap.

These sources support showing strong Python, measurable ML, evaluation and backend skills. They do not establish that one portfolio project will cause an offer.

## Directions and tradeoffs

| Direction | Why it matters | Serious portfolio evidence | Main limitation |
|---|---|---|---|
| Retrieval and grounded AI | Many AI systems depend on finding relevant information before answering. | Compare keyword, dense and hybrid retrieval on independent relevance labels; report quality, latency and failure cases. | Retrieval metrics alone do not establish answer correctness or safety. |
| Agent reliability | Production workflows need checks for incorrect tool use and task failure. | Evaluate real tasks, baseline against simpler workflows, report success, cost and latency. | Deterministic traces alone cannot establish end-to-end agent competence. |
| ML ranking and recommendations | Ranking is a concrete ML function in deployed products. | Time-aware splits, popularity baseline, ranking metrics and a reproducible evaluation. | Public offline metrics need not translate to user impact. |
| Efficient inference | CPU/memory/latency tradeoffs are important for deployment. | Measure quality, throughput, memory and hardware against reproducible baselines. | A smaller model is not automatically faster or accurate enough. |
| Multimodal and robotics | Industry research internships work on perception and action. | Well-scoped experiment on a real benchmark, controlled ablations and error analysis. | Hardware, compute, data and degree prerequisites can be substantial. |

## Choice

This repository implements a retrieval evaluation and search workbench. The decision is an inference from the job descriptions and the availability of an independently labeled public benchmark, not a claim that retrieval is the single best future specialty. The [BEIR benchmark](https://github.com/beir-cellar/beir) supplies a standard corpus, queries and relevance judgments, including SciFact. The underlying [SciFact project](https://github.com/allenai/scifact) concerns evidence for scientific claims. We evaluate document retrieval only: this project does not verify scientific claims or provide medical advice.

Dataset rights need separate review before redistribution or commercial use: the [original SciFact license](https://github.com/allenai/scifact/blob/master/LICENSE.md) distinguishes claims and annotations from abstracts, while the [AllenAI Hugging Face dataset card](https://huggingface.co/datasets/allenai/scifact) lists a different license. This repository downloads the BEIR archive from its source rather than redistributing it; that choice does not resolve the discrepancy.

Future extensions that could make the project stronger include a second dataset or out-of-domain test, explicitly labeled unanswerable queries, calibrated abstention, permission-aware retrieval, user interviews, and an independently judged answer-generation evaluation. They are future work unless implemented and measured. A complementary next project could investigate real recommendation data and time-aware ranking; a separate inference project could measure quantization tradeoffs on a public task. Choose one based on target roles and available compute.
