# Demo data

The two public corpora behind the website's demos, bundled so the demos work
on a fresh clone. Both are unmodified copies; their own licences apply, not the
repository's MIT licence. The third demo, the harness corpus with sub-agents,
is this repository's own data and lives in `results/local_subagents/runs/`.

| folder | what it is | source | licence |
|---|---|---|---|
| `tau2_airline/` | `claude-opus-4-5_high_airline_gpt-5.2_4trials.json`: 200 τ²-bench airline conversations by Claude Opus 4.5 (50 tasks, 4 trials), with the benchmark's evaluation of every run | public results bucket `sierra-tau-bench-public`, submission `claude-opus-4-5_sierra_2026-02-26`, mirrored 2026-09-22 with `experiments/mirror_tau.py`; sha256 `babe71cb463bca64030a91df24431cc6e1df799292dec6b8981618f286f2dc5c` | MIT, © Sierra Research ([tau2-bench](https://github.com/sierra-research/tau2-bench)) |
| `agenthallu_magentic_one/` | `000.json` to `093.json`: the 94 Magentic-One runs of AgentHallu, each with the question, the true answer, the agent's answer and a human hallucination label | [github.com/liuxuannan/AgentHallu](https://github.com/liuxuannan/AgentHallu), folder `Magentic_One`, accessed 2026-09-22 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), © the AgentHallu authors |

Cite the original work when you use these files:

- V. Barres, H. Dong, S. Ray, X. Si and K. Narasimhan, *τ²-Bench: Evaluating
  Conversational Agents in a Dual-Control Environment*, arXiv:2506.07982, 2025.
- X. Liu, X. Yang, Z. Li, P. Li and R. He, *AgentHallu: Benchmarking Automated
  Hallucination Attribution of LLM-based Agents*, arXiv:2601.06818, 2026.

Run either one through the pipeline from the command line:

```bash
python -m agentpm.cli report examples/demos/tau2_airline/claude-opus-4-5_high_airline_gpt-5.2_4trials.json --format tau2 -o report_tau2
python -m agentpm.cli report examples/demos/agenthallu_magentic_one --format agenthallu -o report_agenthallu
```
