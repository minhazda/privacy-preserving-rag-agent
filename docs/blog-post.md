# Building Privacy-Preserving AI Agents: A Practical Guide

*How I built a RAG agent that answers questions about my research, calls a live
forecasting model, and never exposes a single real record.*

## Why privacy is the hard part

Most RAG tutorials stop at "embed documents, retrieve, prompt an LLM." That's
the easy 80%. The hard 20% — the part that decides whether a system can touch
real data — is making privacy a structural property rather than a disclaimer.
This project takes a strict stance: the agent works with **synthetic data and
published research only**, and that guarantee is enforced in code, on every path,
fail-closed.

## The architecture in one breath

A LangGraph ReAct agent sits behind FastAPI with three tools: retrieve research
passages from a local ChromaDB index, run a live demand forecast (calling a
separate forecasting service), and explain a named technique. Every tool result
and every final answer passes through a privacy guard before it reaches the user.

Two choices matter most:

1. **On-device embeddings.** Retrieval uses an ONNX MiniLM model that runs
   locally, so document text never leaves the machine to be embedded.
2. **Framework-agnostic tools.** The tools are plain Python functions, unit
   tested without an LLM; the agent module merely adapts them into LangChain
   tools. The privacy-critical code is therefore exercised by fast, deterministic
   tests — not mocked around.

## Four privacy layers

**1. Redaction + a synthetic-only allow-list.** The guard strips PII (emails,
phone numbers, etc.) and validates that any data flowing through forecasting is
synthetic. Ambiguity is treated as a failure: the guard is *fail-closed*, so a
record it can't confidently classify is blocked, not leaked.

**2. Differential privacy on outputs.** Forecast values can be perturbed with the
Laplace mechanism: clip to bound sensitivity, then add `Laplace(sensitivity / ε)`
noise. Smaller ε means more noise and more privacy. Because the underlying data is
already synthetic, this is opt-in defence-in-depth — but it's a real, tunable
ε-DP primitive.

**3. A tamper-evident audit log.** Every query, tool call, and answer is written
to an append-only JSONL trail. Each entry hashes the previous one, so deleting or
editing any line breaks the chain — and integrity is checkable with one method.
All text is redacted *before* it's written, so the audit log itself stores no PII.

**4. Privacy-safe streaming.** Token streaming is great UX and a privacy hazard:
stream raw tokens and your filter runs too late. The fix is *filter-then-stream* —
produce the fully filtered answer first, then stream it in chunks. Unfiltered
model tokens never hit the wire.

## Agentic patterns that earned their keep

- **Planning via ReAct.** The agent decides whether a question needs retrieval, a
  forecast, an explanation, or several steps — rather than hard-coding a pipeline.
- **Query rewriting.** A deterministic pass expands acronyms (MAE → mean absolute
  error) before retrieval, so dense search matches both forms.
- **Confidence-gated honesty.** If the best retrieved passage is too far from the
  query, the tool returns an explicit "I don't have this" signal instead of
  letting the model improvise. Saying "I don't know" is a feature.
- **Per-session memory.** A LangGraph checkpointer keyed on a session id gives the
  agent real conversational memory; a separate store powers a `/history` endpoint.

## Evaluating without an oracle

Production RAGAS uses an LLM judge, which means cost, latency, and an API key in
CI. To gate quality on every commit, I built deterministic **lexical proxies** for
faithfulness, answer-relevance, and context-precision. They're honestly labelled
as proxies — but they run offline in milliseconds, score a research gold set, and
fail the build if a mean metric drops below threshold. The LLM-judge backend is a
drop-in upgrade for nightly runs.

## What I'd tell someone starting out

- Decide your privacy invariant first, then make it structural and fail-closed.
- Keep the privacy-critical logic in pure functions you can test exhaustively.
- Make the system start even when the LLM can't — degrade, don't crash.
- Treat "I don't know" and the audit log as features, not afterthoughts.
- Stream filtered output, never raw tokens.

## Cost & latency (rough, self-hosted)

Embeddings and retrieval are local and effectively free. Cost is dominated by LLM
tokens per answered query; tool calls add round-trips but no model cost. Latency
is retrieval (single-digit ms locally) plus one or more LLM calls. The exact
numbers depend on the model tier and how many tool hops a question needs —
measure per deployment rather than trusting a table.

## Closing

Privacy-preserving and *useful* aren't in tension if privacy is designed in from
the first commit. The result is an agent you could actually point at sensitive
data — because by construction, it never sees any.

---

*Repo: `github.com/minhazda/privacy-preserving-rag-agent`. Built as Project 2 of a
portfolio; it consumes the forecasting API from Project 1.*
