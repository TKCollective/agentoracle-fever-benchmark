#!/usr/bin/env python3
"""Run AgentOracle + GPT-4o baseline benchmark on 200 FEVER claims."""
import json
import time
import sys
import os
import urllib.request
import urllib.error
import ssl
from datetime import datetime

CLAIMS_FILE = "/home/user/workspace/benchmark/benchmark_claims.jsonl"
RESULTS_FILE = "/home/user/workspace/benchmark/benchmark_results.jsonl"
LOG_FILE = "/home/user/workspace/benchmark/benchmark.log"

AGENTORACLE_URL = "https://agentoracle.co/evaluate"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

DELAY_BETWEEN_CALLS = 2  # seconds
REQUEST_TIMEOUT = 150

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def log(msg):
    ts = datetime.now().isoformat()
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def map_ao_verdict_to_fever(verdict_str, recommendation=None):
    """AgentOracle verdicts: supported, refuted, unverifiable | recommendation: ACT/VERIFY/REJECT."""
    if not verdict_str:
        return "NOT ENOUGH INFO"
    v = verdict_str.lower().strip()
    if v in ("supported", "support", "act"):
        return "SUPPORTS"
    if v in ("refuted", "refute", "reject"):
        return "REFUTES"
    return "NOT ENOUGH INFO"


def call_agentoracle(claim):
    """Returns (fever_verdict, confidence, raw_verdict, elapsed_ms, adversarial_tag, error)."""
    start = time.time()
    payload = json.dumps({"content": claim, "source": "benchmark-fever"}).encode()
    req = urllib.request.Request(AGENTORACLE_URL, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "AgentOracle-Benchmark/1.0")
    try:
        resp = urllib.request.urlopen(req, context=ctx, timeout=REQUEST_TIMEOUT)
        elapsed_ms = int((time.time() - start) * 1000)
        data = json.loads(resp.read())
        # Response structure: {evaluation: {claims: [...], recommendation: ..., overall_confidence: ...}}
        evaluation = data.get("evaluation", {})
        claims = evaluation.get("claims", []) or data.get("claims", [])
        if not claims:
            return ("ERROR", 0, "no_claims", elapsed_ms, None, "no claims returned")
        c = claims[0]
        verdict = c.get("verdict", "")
        confidence = c.get("confidence", 0)
        recommendation = evaluation.get("recommendation", None)
        # Adversarial tag — check explicit field first, then sources map
        adv_tag = c.get("adversarial_result", "")
        if not adv_tag:
            sources = c.get("sources", {})
            adv = sources.get("adversarial", {}) if isinstance(sources, dict) else {}
            adv_verdict = adv.get("verdict", "").lower() if isinstance(adv, dict) else ""
            adv_tag = "contradicted" if adv_verdict in ("refuted", "reject") else "resistant"
        fever_v = map_ao_verdict_to_fever(verdict, recommendation)
        return (fever_v, confidence, verdict, elapsed_ms, adv_tag, None)
    except urllib.error.HTTPError as e:
        elapsed_ms = int((time.time() - start) * 1000)
        return ("ERROR", 0, None, elapsed_ms, None, f"HTTP {e.code}: {e.read()[:200].decode('utf-8', 'ignore')}")
    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        return ("ERROR", 0, None, elapsed_ms, None, f"{type(e).__name__}: {str(e)[:200]}")


def call_gpt4o(claim):
    """GPT-4o baseline via OpenRouter."""
    start = time.time()
    payload = json.dumps({
        "model": "openai/gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": f'Is the following claim true, false, or uncertain based on your knowledge? Claim: "{claim}". Answer with exactly one word: SUPPORTS, REFUTES, or NOT ENOUGH INFO.'
            }
        ],
        "max_tokens": 20,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(OPENROUTER_URL, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {OPENROUTER_KEY}")
    try:
        resp = urllib.request.urlopen(req, context=ctx, timeout=60)
        elapsed_ms = int((time.time() - start) * 1000)
        data = json.loads(resp.read())
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
        # Normalize
        if "SUPPORT" in content or content.startswith("SUPPORTS"):
            v = "SUPPORTS"
        elif "REFUTE" in content or content.startswith("REFUTES"):
            v = "REFUTES"
        elif "NOT ENOUGH" in content or "UNCERTAIN" in content or "NEI" in content:
            v = "NOT ENOUGH INFO"
        else:
            v = "NOT ENOUGH INFO"
        return (v, elapsed_ms, content, None)
    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        return ("ERROR", elapsed_ms, None, f"{type(e).__name__}: {str(e)[:200]}")


def already_processed():
    """Return set of claim_ids already in results file."""
    done = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["claim_id"])
                except:
                    pass
    return done


def main():
    # Load claims
    claims = []
    with open(CLAIMS_FILE) as f:
        for line in f:
            claims.append(json.loads(line))
    log(f"Loaded {len(claims)} claims")

    # Resume support
    done = already_processed()
    if done:
        log(f"Resuming: {len(done)} already processed")
    remaining = [c for c in claims if c["claim_id"] not in done]
    log(f"Processing {len(remaining)} remaining claims")

    correct_ao = 0
    correct_gpt = 0
    errors_ao = 0
    errors_gpt = 0
    processed_session = 0

    with open(RESULTS_FILE, "a") as out:
        for i, claim in enumerate(remaining):
            claim_id = claim["claim_id"]
            text = claim["claim_text"]
            truth = claim["ground_truth"]

            # AgentOracle
            ao_v, ao_conf, ao_raw, ao_ms, adv_tag, ao_err = call_agentoracle(text)
            ao_correct = ao_v == truth

            # Small delay between APIs
            time.sleep(0.5)

            # GPT-4o baseline
            gpt_v, gpt_ms, gpt_raw, gpt_err = call_gpt4o(text)
            gpt_correct = gpt_v == truth

            result = {
                "claim_id": claim_id,
                "claim_text": text,
                "ground_truth": truth,
                "agentoracle": {
                    "verdict_fever": ao_v,
                    "verdict_raw": ao_raw,
                    "confidence": ao_conf,
                    "elapsed_ms": ao_ms,
                    "correct": ao_correct,
                    "adversarial_tag": adv_tag,
                    "error": ao_err,
                },
                "gpt4o": {
                    "verdict_fever": gpt_v,
                    "verdict_raw": gpt_raw,
                    "elapsed_ms": gpt_ms,
                    "correct": gpt_correct,
                    "error": gpt_err,
                },
            }
            out.write(json.dumps(result) + "\n")
            out.flush()

            if ao_err:
                errors_ao += 1
            elif ao_correct:
                correct_ao += 1
            if gpt_err:
                errors_gpt += 1
            elif gpt_correct:
                correct_gpt += 1
            processed_session += 1

            if (processed_session) % 10 == 0:
                total_so_far = len(done) + processed_session
                log(f"Progress: {total_so_far}/{len(claims)} | AO correct: {correct_ao}, errors: {errors_ao} | GPT-4o correct: {correct_gpt}, errors: {errors_gpt}")

            # Rate limit
            time.sleep(DELAY_BETWEEN_CALLS)

    log(f"DONE. Session processed: {processed_session}")
    log(f"Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
