# ACL-Style Final Report Outline for ZeroShotVDR

This document organizes the final paper structure for the ZeroShotVDR course report.
It is written for an ACL-style paper with 8-9 pages of main content in English.
References and appendix are assumed to be outside the page limit.

## 1. Paper Positioning

Recommended paper framing:

- Task: zero-shot visual document retrieval on MMLongBench DocumentQA.
- Baseline: ColPali page retrieval with late interaction.
- Main engineering correction: stable page identity and query-specific candidate scope.
- Main methodological contribution: query-adaptive two-stage coarse-to-fine retrieval.
- Final recommended system: adaptive top-N + neighbor expansion + mean-pool cache.
- Main claim: the proposed method preserves or slightly improves retrieval quality while reducing latency, especially on long candidate sets.

What the paper should not claim:

- Do not present direction A (adaptive index compression) as a completed contribution.
- Do not describe the method as a global corpus retriever; the main evaluation protocol is query-scoped candidate retrieval.
- Do not claim uniform gains on every slice; the real conclusion is a quality-efficiency trade-off with especially strong gains on long-context SlideVQA slices.

## 2. Candidate Titles

Choose one of the following styles.

1. ZeroShotVDR: Query-Adaptive Two-Stage Retrieval for Zero-Shot Visual Document Retrieval
2. Efficient Zero-Shot Visual Document Retrieval with Adaptive Coarse-to-Fine Page Ranking
3. Improving ColPali-Based Page Retrieval with Adaptive Candidate Selection and Neighbor Expansion

The first title is the safest because it keeps the project identity and states the main method clearly.

## 3. Recommended Page Budget

| Section | Target Pages |
| --- | ---: |
| Abstract | 0.2 |
| 1 Introduction | 1.0 - 1.2 |
| 2 Related Work | 0.8 - 1.0 |
| 3 Task and Method | 2.0 - 2.3 |
| 4 Experimental Setup | 1.2 - 1.5 |
| 5 Results | 1.4 - 1.7 |
| 6 Analysis and Discussion | 1.0 - 1.2 |
| 7 Conclusion | 0.2 - 0.4 |
| Total | 7.8 - 9.5 |

If the draft becomes too long, merge part of Section 6 into Section 5 and keep the main body close to 8.5 pages.

## 4. Recommended Section Structure

## Title

Use one of the candidate titles above.

## Abstract

Target length: 150-200 words.

Recommended sentence plan:

1. Introduce the task: zero-shot visual document retrieval for visually rich long documents is challenging because OCR-only pipelines lose layout and visual evidence.
2. Introduce the baseline and limitation: ColPali is a strong OCR-free late-interaction baseline, but full MaxSim reranking becomes inefficient and less robust on long candidate sets.
3. State the method: propose a query-adaptive two-stage retrieval pipeline with mean-pool coarse retrieval, adaptive candidate selection, optional neighbor expansion, and a compact mean-pool cache.
4. State the setting: evaluate on MMLongBench DocumentQA under a cleaned valid-only protocol with 14,385 queries.
5. State the headline result: the final method reaches Recall@10 = 0.8523 and nDCG@10 = 0.6325, while reducing average latency from 0.0716 s/query to 0.0600 s/query and P95 latency from 0.1384 s to 0.0858 s.
6. End with the interpretation: gains are strongest on long-context SlideVQA slices, showing that adaptive coarse-to-fine retrieval is an effective quality-efficiency improvement over the ColPali baseline.

## 1 Introduction

Target length: about 1 page.

Recommended paragraph plan:

Paragraph 1: task motivation.
- Explain why visually rich documents are difficult for OCR-centric retrieval.
- Mention layouts, charts, tables, figures, and multi-region pages.

Paragraph 2: baseline opportunity and challenge.
- Introduce ColPali as an OCR-free page retrieval approach using late interaction between query tokens and page patches.
- Explain that the method is effective but expensive when the candidate set becomes large.

Paragraph 3: project-specific challenge on MMLongBench.
- Introduce MMLongBench DocumentQA with three subtasks and long context lengths up to K128.
- State that the hardest cases concentrate on large candidate sets and visually similar neighboring pages.

Paragraph 4: your approach.
- Summarize the two-stage pipeline: mean-pooled coarse retrieval, adaptive top-N selection, neighbor expansion, and full MaxSim reranking.
- Mention that the method is built on top of a corrected stable evaluation protocol.

Paragraph 5: contributions.
- Use a short bullet list with 3 contributions.

Suggested contribution bullets:

- We build a stable ColPali-based page retrieval pipeline for MMLongBench DocumentQA, including stable page identity reconstruction and a valid-only evaluation protocol that removes queries with unusable page-level supervision.
- We propose a query-adaptive two-stage retrieval method that combines mean-pool coarse retrieval, adaptive candidate sizing, and neighbor-aware expansion before full MaxSim reranking.
- We show that the final system maintains baseline quality while improving efficiency, with the largest gains appearing on long-context SlideVQA slices.

## 2 Related Work

Target length: 0.8-1.0 pages.

Recommended subsection flow:

### 2.1 OCR-Free Visual Document Retrieval
- Discuss ColPali as the primary reference.
- Briefly contrast with OCR-plus-text-index pipelines.
- Emphasize that your work is an efficiency-oriented extension on top of OCR-free page retrieval.

### 2.2 Late Interaction Retrieval
- Discuss ColBERT as the conceptual origin of late interaction and MaxSim.
- Explain why late interaction preserves fine-grained matching but increases scoring cost.

### 2.3 Long-Context Vision-Language Evaluation
- Introduce MMLongBench as the benchmark context.
- Position DocumentQA as the most relevant subset for page retrieval.

Keep this section compact. The goal is to justify the design space, not to survey all document understanding work.

## 3 Task and Method

Target length: 2.0-2.3 pages.

This should be the core technical section.

### 3.1 Task Definition and Evaluation Protocol
- Define page retrieval: given a text query, rank pages from the query-specific candidate set.
- Explain the ground truth source: ans_page_list in DocumentQA.
- State the query-scoped candidate protocol clearly.
- State why the paper uses the valid-only subset of 14,385 queries for main comparisons.
- Briefly explain the page identity correction: page IDs are reconstructed from source image paths instead of sample-local enumeration.

This subsection matters because it turns the paper from a simple engineering report into a defensible experimental study.

### 3.2 Stable Baseline
- Describe the five-layer pipeline briefly: data adapter, indexing, retrieval, evaluation, configuration support.
- Explain page encoding with ColPali and per-page patch embeddings.
- Explain query encoding and MaxSim reranking.
- Mention per-page storage and query-specific candidate scope.

Include the MaxSim formula:

$$
Score(Q, P) = \sum_i \max_j \text{Sim}(q_i, p_j)
$$

State that Sim is the dot product after L2 normalization.

### 3.3 Query-Adaptive Two-Stage Retrieval
- Motivate from Step 3 findings: hardest slices have large candidate sets and neighbor confusion.
- Describe stage 1: mean-pool coarse retrieval over candidate pages.
- Describe stage 2: rerank only a reduced candidate set with full MaxSim.
- Describe adaptive top-N selection rather than fixed top-N.
- Describe neighbor expansion and why it helps multi-page or adjacent-page evidence.

Recommended subsection breakdown:

#### 3.3.1 Mean-Pool Coarse Retrieval
- Mean-pool page patch embeddings into one page vector.
- Rank candidate pages quickly in this low-cost view.

#### 3.3.2 Adaptive Candidate Selection
- Explain that candidate size depends on query-specific score distribution and bounded min/max candidate counts.
- Contrast briefly with fixed top-32, top-64, and top-128 baselines.

#### 3.3.3 Neighbor Expansion
- Expand around top-ranked seed pages with a small page window.
- Motivate with multi-page evidence and neighboring-page confusion.

### 3.4 Mean-Pool Cache
- Describe the cache as a precomputed mean-pooled view for all pages.
- Emphasize that it changes efficiency, not ranking quality.
- Mention the storage comparison: about 26 MB cache versus about 88.6 GB patch index.

### 3.5 Complexity and Expected Benefits
- Explain qualitatively how reducing rerank candidates lowers latency.
- State that the method is most valuable when the candidate universe is large.

## 4 Experimental Setup

Target length: 1.2-1.5 pages.

### 4.1 Dataset and Scope
- Dataset: MMLongBench DocumentQA.
- Subtasks: longdocurl, mmlongdoc, slidevqa.
- Length buckets: K4, K8, K16, K32, K64, K128.
- Main evaluation subset: 14,385 valid-only queries.

### 4.2 Metrics
- Recall@k, Precision@k, MRR, nDCG@k.
- For the main paper, focus tables on Recall@10, nDCG@10, MRR, average latency, P95 latency, and average rerank candidates.

### 4.3 Baselines and Ablations
- Phase 3 full MaxSim baseline.
- Fixed Top-32.
- Fixed Top-64.
- Fixed Top-128.
- Adaptive.
- Adaptive + Neighbor.
- Cache variants for Fixed Top-64, Adaptive, and Adaptive + Neighbor.

### 4.4 Implementation Details
- ColPali-v1.3 with its base model.
- Hardware: Ubuntu + 2x RTX 3090 24 GB.
- Index size and run scale.
- Briefly mention separate indexing and retrieval stages.

Avoid drowning this section in environment details. Keep only what supports reproducibility.

## 5 Results

Target length: 1.4-1.7 pages.

### 5.1 Main Results
- Present the main ablation table.
- Center the comparison on Phase 3 baseline versus Adaptive + Neighbor + Cache.
- Use exact numbers already fixed by the project documents.

Recommended interpretation:

- The final method improves Recall@10 slightly from 0.8517 to 0.8523.
- nDCG@10 stays at 0.6325.
- Average latency drops from 0.0716 s/query to 0.0600 s/query.
- P95 latency drops from 0.1384 s to 0.0858 s.

### 5.2 Fixed Top-N vs Adaptive Selection
- Compare fixed top-32, fixed top-64, fixed top-128, and adaptive.
- Explain that top-32 is too aggressive, top-128 recovers quality but hurts latency, and adaptive finds a better balance.

### 5.3 Effect of Neighbor Expansion
- Compare Adaptive versus Adaptive + Neighbor.
- Explain that neighbor expansion is the main quality recovery mechanism that offsets losses from aggressive pruning.

### 5.4 Effect of Mean-Pool Cache
- Show that cache variants preserve quality but substantially reduce latency.
- State clearly that the cache is a deployment-oriented optimization.

## 6 Analysis and Discussion

Target length: 1.0-1.2 pages.

### 6.1 Why the Method Helps
- Use the slice results to show strongest improvements on slidevqa/K64 and slidevqa/K128.
- Explain that these slices have large candidate sets and visually repetitive layouts, making coarse filtering especially useful.

### 6.2 Where the Trade-Off Appears
- Discuss the mild regressions on longdocurl/K128 and mmlongdoc/K128.
- Interpret them as cases where reducing candidate space can drop some fine-grained evidence.
- Be honest and precise here. This increases paper credibility.

### 6.3 Failure Modes
- Reuse the three main failure types from Phase 3 analysis: miss_top10, multi-page partial recall, neighbor-page confusion.
- Explain which ones are reduced and which remain open.

### 6.4 Limitations
- The evaluation uses query-scoped candidate retrieval rather than fully global retrieval.
- Direction A was not completed.
- Some DocumentQA queries have unusable page-level supervision, so cleaned evaluation is necessary.

## 7 Conclusion

Target length: 0.2-0.4 pages.

Recommended conclusion structure:

1. Restate the problem.
2. Restate the solution in one sentence.
3. Restate the main empirical takeaway.
4. End with future work: stronger global retrieval and storage-aware compression.

## 5. Recommended Figures and Tables

Keep the main paper visual inventory compact and purposeful.

### Figures

Figure 1: Overall system diagram.
- Stable baseline pipeline plus Phase 4 two-stage branch.
- This should be the main method figure.

Figure 2: Slice-level comparison or K128 bucket comparison.
- Prefer a compact bar chart showing Phase 3 vs final method on the five key slices.

Optional Figure 3: Latency-quality trade-off plot.
- X-axis: average latency.
- Y-axis: Recall@10 or nDCG@10.
- Points: baseline, fixed top-32, fixed top-64, fixed top-128, adaptive, adaptive + neighbor, cache variants.

### Tables

Table 1: Dataset and evaluation scope.
- Subtasks, length buckets, number of queries, valid-only protocol.

Table 2: Main ablation results.
- The most important table in the paper.

Table 3: Cache comparison.
- Show no-cache versus cache for the main methods.

Table 4: Slice-level analysis.
- Use the five slices already highlighted in the milestone report.

## 6. Suggested Appendix Contents

Move the following to appendix if space is tight:

- Full results for all k values.
- Full metrics by subtask and by length.
- Additional bad-case examples.
- Full command-line and implementation details.
- Extra notes on stable page ID reconstruction and invalid ground truth filtering.

## 7. Writing Priorities

If time is limited, prioritize these elements first:

1. Abstract
2. Introduction with contributions
3. Method figure and Section 3
4. Main result table
5. Slice analysis table
6. Conclusion

Only then expand related work and appendix.

## 8. Recommended English Tone

Use the following writing style throughout the paper:

- Precise and conservative.
- Prefer "improves efficiency while preserving quality" over "significantly outperforms".
- Explicitly mention trade-offs instead of hiding them.
- Use "valid-only evaluation protocol" consistently when describing the main comparison.

Recommended one-sentence summary for the introduction and conclusion:

"We build a stable ColPali-based page retrieval baseline for MMLongBench DocumentQA and show that a query-adaptive coarse-to-fine reranking pipeline with neighbor expansion and a mean-pool cache improves the quality-efficiency trade-off, especially on long candidate sets." 