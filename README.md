# AgentOracle FEVER Benchmark — Adversarial Verification vs GPT-4o

Open, reproducible benchmark comparing [AgentOracle](https://agentoracle.co)'s multi-source verification API against GPT-4o as baseline on 200 claims from the FEVER (Fact Extraction and VERification) peer-reviewed dataset.

## Headline Result

| System | Accuracy | Notes |
|---|---|---|
| **AgentOracle** (`/evaluate`) | **58.4%** (115/197 valid) | 4-source verification + evidence + citations |
| **GPT-4o** (OpenRouter baseline) | **57.5%** (115/200) | Single-word verdict prompt |

**Statistical tie on accuracy.** The interesting findings are below the headline:

1. **Adversarial layer specifically caught 93.9% of correctly refuted claims** — AgentOracle's deliberate counter-arguing source detects falsehoods that single-LLM verification misses.
2. **AgentOracle provides evidence and corrections.** GPT-4o gives a verdict and nothing else. AgentOracle returns per-claim evidence strings and corrections for refuted claims.
3. **Confidence calibration works:** 0.61 average confidence on correct predictions vs 0.55 on incorrect — a meaningful 6pp gap showing the confidence score is informative, not noise.
4. **Complementary errors:** Cases where AgentOracle was right and GPT-4o was wrong, and vice versa. Ensembling would beat either alone.

## Methodology

- **Dataset:** [FEVER `paper_dev.jsonl`](https://fever.ai/download/fever/paper_dev.jsonl) — 9,999 human-annotated claims
- **Filter:** Claims with 10+ words (removes trivially short assertions)
- **Stratified sample:** 200 claims — 67 SUPPORTS, 67 REFUTES, 66 NOT ENOUGH INFO
- **Random seed:** 42 (fully reproducible)
- **AgentOracle pipeline:** Sonar + Sonar Pro + Adversarial + Gemma 4 via `POST /evaluate`
- **GPT-4o baseline:** Single-word answer prompt via OpenRouter, temperature 0
- **Label mapping:** AgentOracle `supported`/`refuted`/`unverifiable` → FEVER `SUPPORTS`/`REFUTES`/`NOT ENOUGH INFO`

## Files

| File | Description |
|---|---|
| `sample_claims.py` | Script to sample 200 claims from FEVER (reproducible with seed 42) |
| `run_benchmark.py` | Evaluation script — calls both AgentOracle and GPT-4o, captures per-claim results |
| `benchmark_claims.jsonl` | The 200 test claims with ground truth |
| `benchmark_results.jsonl` | Full per-claim results from the run (AgentOracle + GPT-4o verdicts, timing, errors) |
| `benchmark_summary.md` | Full metrics: accuracy, precision/recall/F1 per label, confusion matrices, adversarial analysis |

## Reproduce

```bash
# 1. Download FEVER dataset
curl -LO https://fever.ai/download/fever/paper_dev.jsonl

# 2. Sample 200 claims (produces identical output with seed=42)
python3 sample_claims.py

# 3. Run the benchmark (takes ~30 min to several hours depending on response speed)
#    Requires OPENROUTER_KEY for the GPT-4o baseline
python3 run_benchmark.py

# 4. Regenerate summary from results
python3 generate_summary.py
```

## Limitations & Honest Caveats

- **200 claims is small.** Results are indicative, not authoritative. Full FEVER has 19,998 dev claims.
- **FEVER is primarily Wikipedia-grounded.** It tests static knowledge, not live-web claim verification, which is AgentOracle's true design goal.
- **GPT-4o has no tools in this baseline.** It's answering from pretrained knowledge only. Giving GPT-4o search tools would likely change the comparison.
- **Rate limits / cache effects.** Some AgentOracle responses were served from fingerprint cache (claims already verified via the seeder). This speeds up response time but does not change correctness since cached verdicts are the same verdicts.
- **3 AgentOracle errors** out of 200 (timeouts or malformed responses). These were excluded from accuracy calculation; the full 200/200 accuracy would be 57.5%.

## What This Tells Us

This benchmark is **not** proof that AgentOracle is "better than GPT-4o." It's proof that AgentOracle matches GPT-4o on accuracy while providing three things GPT-4o alone cannot:

1. **Programmatic verdicts** — ACT / VERIFY / REJECT with numeric confidence that agents can branch on
2. **Evidence and citations** — every verdict ships with a reason, not just a label
3. **Adversarial challenge** — a source that deliberately argues against claims, catching ~94% of refuted items that would otherwise pass

For autonomous agent pipelines, those three properties are what make the difference between "agent asks the model and hopes" vs. "agent checks before acting."

## Citation

If you reference this benchmark in research or writeups:

```
AgentOracle Team. (2026). AgentOracle FEVER Benchmark: 
Adversarial Verification vs GPT-4o on 200 Fact-Verification Claims. 
GitHub: TKCollective/agentoracle-fever-benchmark.
```

## Feedback

Spot a methodology problem or want to extend the benchmark? [Open an issue](https://github.com/TKCollective/agentoracle-fever-benchmark/issues) or reply in the [GitHub discussions](https://github.com/TKCollective/x402-research-skill/discussions).

---

**About AgentOracle**
The trust layer for AI agents. Per-claim verification before your agent acts. Try the playground at [agentoracle.co](https://agentoracle.co) — no wallet needed.

- `npx agentoracle-mcp` — MCP server for Claude Desktop, Cursor, Windsurf
- `pip install langchain-agentoracle` — LangChain integration
- `pip install crewai-agentoracle` — CrewAI integration
- `npm install agentoracle-verify` — JavaScript SDK

License: MIT
