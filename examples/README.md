# Examples

Small inputs to try the pipeline on without downloading anything. The tests use
most of them.

| file | format | what it shows |
|---|---|---|
| `sample_trace.json` | `generic` | a run that retrieves before it answers; passes `reference_model.json` |
| `sample_trace_faulted.json` | `generic` | a run that reports success but answers before retrieving; fails three constraints |
| `reference_model.json` | reference | retrieve before answer, nothing after the answer, end on the answer |
| `openai_spans.jsonl` | `openai` | an OpenAI Agents SDK export with a handoff between two agents |
| `openai_demo.ocel.json` | OCEL 2.0 | `openai_spans.jsonl` turned into an object-centric log |
| `openai_capture.py` | | the trace processor that records such an export from your own SDK application |
| `claude_transcript_sample.jsonl` | `claude` | a hand-made Claude Code transcript: Read, Edit, Bash, and a sub-agent's Grep |
| `claude_reference.json` | reference | a Claude Code editing protocol (Read before Edit); the sample is flagged because it ends on the sub-agent's Grep |
| `task_types_tau2_airline.json` | task types | the template from which every τ²-bench airline run gets the model of its task type (`--task-types`), as in the thesis |
| `task_types_tau2_retail.json` | task types | the same for the retail domain; it gives the models of `experiments/retail_task_types.py`, which were frozen before any retail run was scored (a test checks this) |
| `tau2_airline_task40_trial1.json` | `tau2` | trial 1 of airline task 40 by Claude Opus 4.5, cut out of the bundled demo file by `experiments/example_run.py`: the run that Appendix A of the thesis follows through every layer |
| `demos/` | | the two public corpora behind the website's demos, see `demos/README.md` |

```bash
python -m agentpm.cli run examples/sample_trace.json         --format generic --reference examples/reference_model.json
python -m agentpm.cli run examples/sample_trace_faulted.json --format generic --reference examples/reference_model.json -v
python -m agentpm.cli ocel examples/openai_spans.jsonl --format openai -o openai.ocel.json
python -m agentpm.cli run examples/claude_transcript_sample.jsonl --format claude --reference examples/claude_reference.json -v
python -m agentpm.cli report examples/tau2_airline_task40_trial1.json --format tau2 --task-types examples/task_types_tau2_airline.json -o report_example
```
