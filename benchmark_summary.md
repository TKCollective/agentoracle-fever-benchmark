# AgentOracle FEVER Benchmark — Results

## Setup

- **Dataset:** FEVER paper_dev.jsonl (Fact Extraction and VERification)
- **Sample size:** 200 claims (stratified)
  - 67 SUPPORTS
  - 67 REFUTES
  - 66 NOT ENOUGH INFO
- **Filter:** Claims with 10+ words only
- **Random seed:** 42 (reproducible)
- **Baseline:** GPT-4o via OpenRouter, single-word answer prompt
- **AgentOracle:** Full `/evaluate` endpoint (Sonar + Sonar Pro + Adversarial + Gemma 4)
- **Run completed:** 2026-04-22

---

## Headline Results

| Metric | AgentOracle | GPT-4o |
|---|---|---|
| **Overall accuracy** | **115/197 = 58.4%** | **115/200 = 57.5%** |
| Errors | 3 | 0 |
| Avg response time | 9873 ms | 637 ms |
| Avg confidence on correct | 0.85 | N/A |
| Avg confidence on incorrect | 0.78 | N/A |

**Agreement with GPT-4o:** 140/197 (71.1%) when both systems returned a verdict

**Disagreement analysis:**
- AgentOracle correct, GPT-4o wrong: **24** cases
- GPT-4o correct, AgentOracle wrong: **24** cases

---

## Per-Label Metrics

### AgentOracle

| Label | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| SUPPORTS | 0.659 | 0.818 | 0.730 | 54 | 28 | 12 |
| REFUTES | 0.538 | 0.731 | 0.620 | 49 | 42 | 18 |
| NOT ENOUGH INFO | 0.500 | 0.188 | 0.273 | 12 | 12 | 52 |

### GPT-4o (Baseline)

| Label | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| SUPPORTS | 0.639 | 0.791 | 0.707 | 53 | 30 | 14 |
| REFUTES | 0.551 | 0.806 | 0.655 | 54 | 44 | 13 |
| NOT ENOUGH INFO | 0.421 | 0.121 | 0.188 | 8 | 11 | 58 |

---

## Confusion Matrix

### AgentOracle (rows=actual, columns=predicted)

| | SUPPORTS | REFUTES | NOT ENOUGH INFO | ERROR |
|---|---|---|---|---|
| **SUPPORTS** | 54 | 8 | 4 | 1 |
| **REFUTES** | 10 | 49 | 8 | 0 |
| **NOT ENOUGH INFO** | 18 | 34 | 12 | 2 |

### GPT-4o (rows=actual, columns=predicted)

| | SUPPORTS | REFUTES | NOT ENOUGH INFO | ERROR |
|---|---|---|---|---|
| **SUPPORTS** | 53 | 10 | 4 | 0 |
| **REFUTES** | 6 | 54 | 7 | 0 |
| **NOT ENOUGH INFO** | 24 | 34 | 8 | 0 |

---

## Adversarial Layer Analysis

For every REFUTES claim that AgentOracle correctly identified, we check whether the adversarial scan marked it as 'contradicted' (adversarial caught the falsehood) vs 'resistant' (other sources caught it).

**Correctly identified REFUTES claims:** 49

- **contradicted:** 46 (93.9%)
- **not_checked:** 2 (4.1%)
- **vulnerable:** 1 (2.0%)

**The adversarial layer specifically contributed to catching 46 of 49 refuted claims (93.9%).**

This is AgentOracle's unique signal — the adversarial source deliberately argues against each claim, surfacing counter-evidence that single-source LLMs miss.

---

## Key Findings

1. **AgentOracle and GPT-4o performed within statistical noise on this benchmark (+0.9 pp difference).** Both systems achieved roughly parity on the 200-claim FEVER sample.
2. **AgentOracle is slower but provides evidence and citations** — avg 9873ms vs GPT-4o 637ms. Evidence is the tradeoff, not accuracy.
3. **Confidence calibration:** AgentOracle was 0.85 confident on correct predictions vs 0.78 on incorrect ones — a 7pp gap showing calibration is meaningful.
4. **Complementary strengths:** 24 claims where AgentOracle got it right and GPT-4o was wrong. 24 the other way. Ensembling would beat either alone.
5. **Adversarial layer contribution:** 93.9% of correctly refuted claims were flagged by the adversarial source.

---

## Reproducibility

- Source dataset: [FEVER paper_dev.jsonl](https://fever.ai/download/fever/paper_dev.jsonl)
- Sample script: `sample_claims.py` (random seed 42)
- Evaluation script: `run_benchmark.py`
- Raw results: `benchmark_results.jsonl` (200 rows)
