================================================================================
README — Hallucination in Large Language Models: A Configuration-Level Study
================================================================================
Author:       Ibrahim Alqubaisi
Student ID:   220022292
Programme:    MSc Artificial Intelligence
Module:       INM434 Natural Language Processing
Institution:  City, University of London
Email:        ibrahim.alqubaisi@city.ac.uk
Deadline:     May 3, 2026

GitHub Repository:
  https://github.com/IQubaisi/hallucination-triviaqa

Google Colab Notebook (shareable link):
  https://colab.research.google.com/drive/1HO5oBaBNg9BkwrTnL2SCU8YiMNpQzFIc?usp=sharing

================================================================================
PROJECT OVERVIEW
================================================================================

This project conducts a controlled empirical comparison of eight LLM
configurations (two model sizes x four prompt strategies) on the full
TriviaQA RC validation set (17,944 questions). It measures Exact Match,
token-level F1, and manually corrected hallucination rates across all
eight configurations, with a calibration analysis as an extension.

The two primary models evaluated are:
  - Llama-3.2-3B-Instruct  (small baseline)
  - Llama-3.1-8B-Instruct  (larger model)

The four configurations are:
  - Config 1: Closed-book, neutral prompt
  - Config 2: Closed-book, abstention prompt
  - Config 3: RAG with BM25 retrieval
  - Config 4: Calibrated confidence prompting

================================================================================
REQUIREMENTS
================================================================================

Hardware:
  - GPU with at least 40GB VRAM (tested on NVIDIA A100-SXM4-80GB)
  - Google Colab Pro with A100 runtime is recommended
  - Minimum 25GB system RAM

Python version:
  - Python 3.12 (as provided by Google Colab)

Required libraries (install via pip):
  transformers
  datasets
  accelerate
  bitsandbytes
  pyserini
  torch
  pillow==10.4.0
  rank-bm25
  pandas
  numpy
  matplotlib
  openpyxl

Install all dependencies by running Section 1 of the notebook:
  !pip install transformers datasets accelerate bitsandbytes pyserini torch pillow==10.4.0 -q
  !pip install rank-bm25 -q

HuggingFace Access:
  - A HuggingFace account with access to Meta Llama models is required.
  - Request access at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
  - Request access at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
  - Store your HuggingFace token as a Colab secret named HF_TOKEN
    (click the key icon in the left sidebar of Colab)

Google Drive:
  - A Google Drive account is required. Results are saved to:
    /content/drive/MyDrive/hallucination_project/
  - The notebook mounts Drive automatically in Section 1.

================================================================================
DATASET
================================================================================

Dataset: TriviaQA RC (Reading Comprehension) validation split
Source:  HuggingFace Datasets — trivia_qa, rc configuration
Size:    17,944 questions
Link:    https://huggingface.co/datasets/trivia_qa

The dataset is downloaded automatically when Section 2 of the notebook runs:
  dataset = load_dataset("trivia_qa", "rc", split="validation")

No manual download is required.

================================================================================
HOW TO RUN THE NOTEBOOK
================================================================================

IMPORTANT: The full inference pipeline takes approximately 4-5 hours on an
A100 GPU. All 8 inference results are already saved to Google Drive and
pre-computed results are available in this submission package. To reproduce
evaluation only (without re-running inference), follow the SHORT RUN below.

--- FULL RUN (reproduces all results from scratch) ---

1. Open hallucination_triviaqa.ipynb in Google Colab
2. Set runtime to A100 GPU (Runtime → Change runtime type → A100)
3. Add HuggingFace token to Colab secrets as HF_TOKEN
4. Run Section 1:  Environment setup and Drive mount
5. Run Section 2:  Load TriviaQA dataset
6. Run Section 3:  Load Llama-3.2-3B-Instruct model
7. Run Section 4:  Define prompt templates
8. Run Section 5:  Define batched inference function
9. Run Section 6:  Config 1 inference (3B) — ~30 minutes
10. Run Section 7:  Save Config 1 3B results to Drive
11. Run Section 8:  Config 2 inference (3B) — ~30 minutes
12. Run Section 9:  Config 4 inference (3B) — ~35 minutes
    [Note: Config 3 is run after BM25 index construction in Section 15]
13. Run Section 10: Load Llama-3.1-8B-Instruct model
14. Run Section 11: Config 1 inference (8B) — ~30 minutes
15. Run Section 12: Config 2 inference (8B) — ~30 minutes
16. Run Section 14: Config 4 inference (8B) — ~35 minutes
17. Run Section 15: Build BM25 corpus and pre-compute retrievals — ~5 minutes
18. Run Section 16: Config 3 inference (8B) — ~30 minutes
19. Run Section 17: Reload 3B model
20. Run Section 18: Config 3 inference (3B) — ~30 minutes
21. Continue with Sections 19-27 for evaluation (see SHORT RUN below)

--- SHORT RUN (evaluation only, uses pre-saved results) ---

All 8 inference results are pre-saved to Google Drive. To reproduce
evaluation, figures, and tables without re-running inference:

1. Open hallucination_triviaqa.ipynb in Google Colab
2. Set runtime to CPU (no GPU needed for evaluation)
3. Mount Google Drive (Section 1 or Section 7)
4. Run Section 1:  Environment setup
5. Run Section 2:  Load dataset (needed for alias scoring)
6. Run Section 4:  Define prompt templates
7. Run Section 5:  Define inference function
8. Run Section 19: Load all 8 results from Drive
9. Run Section 20: Define answer extraction functions
10. Run Section 21: Define EM and F1 scoring functions
11. Run Section 22: Score all 8 runs and apply labels
12. Run Section 23: Generate summary results table
13. Run Section 24: Generate manual annotation sample
14. Run Section 25: Load and analyse manual annotations
15. Run Section 26: Generate final corrected results table
16. Run Section 26b: Config 4 8B format collapse analysis
17. Run Section 27: Generate figures
18. Run Section 28: Push to GitHub (optional)

================================================================================
KEY PARAMETERS
================================================================================

Inference:
  batch_size      = 16
  max_new_tokens  = 50
  decoding        = greedy (do_sample=False)
  precision       = float16
  random_seed     = 42

Retrieval (Config 3):
  retriever       = BM25 (rank-bm25 library)
  corpus_size     = 17,944 passages (one per question)
  top_k           = 1
  passage_truncation = 300 words
  fallback        = none (zero closed-book fallback)

Hallucination labelling:
  Correct         = EM == 1
  Hallucinated    = EM == 0 AND F1 < 0.3
  Uncertain       = EM == 0 AND F1 >= 0.3
  Abstained       = model output matches abstention pattern

Correction factors (from 208 manual annotations):
  Hallucinated precision = 0.865
  Uncertain precision    = 0.075

================================================================================
OUTPUT FILES
================================================================================

All output files are saved to Google Drive under:
  /content/drive/MyDrive/hallucination_project/

  results_config{1-4}_{3b,8b}.json   — raw model outputs (8 files)
  scored_config{1-4}_{3b,8b}.json    — scored results with labels (8 files)
  retrieved_passages.json             — BM25 retrieved passages for Config 3
  final_results.csv                   — corrected hallucination rates table
  summary_results.csv                 — automatic label summary
  manual_annotation_sample2_complete.xlsx — 208 manually annotated cases
  manual_summary_2.csv                — per-config manual rates
  results_figure.png                  — main results figure
  main_results.pdf                    — main results figure (PDF for report)

================================================================================
REPOSITORY CONTENTS
================================================================================

  hallucination_triviaqa.ipynb          — full Colab notebook
  final_results.csv                     — corrected hallucination rates
  summary_results.csv                   — automatic label summary
  manual_annotation_sample2_complete.xlsx — 208 annotated cases
  manual_summary_2.csv                  — per-config manual rates
  results_figure.png                    — main results figure
  readme.txt                            — this file

================================================================================
NOTES
================================================================================

1. Config 3 (RAG) is run AFTER Configs 1, 2, and 4 in the notebook because
   it requires the BM25 index to be built first (Section 15). This is
   intentional and documented in the Section 9 markdown header.

2. The 8B model under Config 4 shows 56.5% unparseable outputs due to
   format compliance collapse. This is documented and analysed in Section 26b.
   The reported hallucination rate for 8B Config 4 conflates format collapse
   with genuine hallucination — see report Section 5.1 for full discussion.

3. Manual annotation was performed by the author on 208 stratified cases.
   Annotation guidelines: Correct = matches gold answer; Hallucinated =
   factually wrong or unsupported; Uncertain = cannot determine from gold
   answer alone; Abstained = model declined to answer.

4. All experiments were run on Google Colab Pro with NVIDIA A100-SXM4-80GB
   GPU under Python 3.12.

================================================================================
