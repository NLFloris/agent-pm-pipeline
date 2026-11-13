# Report: capture.jsonl (otel)

6 runs. Produced by `agentpm report`.

| file | what it is |
|---|---|
| `CHECKS.txt` | what the source exposes, checked automatically; read this first |
| `inventory.txt` | counts, and every run on one line as agent:activity |
| `event_log.csv` | the flat event log, one row per event, with arguments, results and reasoning |
| `runs.ocel.json` | the object-centric log, OCEL 2.0 |
| `lenses.txt` | discovery over the listing and over the steps, handovers and repeated tool use |
| `event_log.xes` | the flat log in XES, for ProM, Disco and pm4py |
| `conformance.txt` | each run against the reference, with the event responsible |
| `verdicts.json` | the same, machine-readable |
| `pm4py_dfg.png, pm4py_petri_net.png, pm4py_process_tree.png, pm4py_bpmn.png` | pm4py's directly-follows graph and inductive-miner model (noise 0.0) |
| `pm4py_summary.txt` | variants, start and end activities, model sizes |
| `handover.png` | who hands work to whom |
| `lane_model.png` | aggregate model, one lane per agent |
| `dfg_by_agent.png` | directly-follows graph, nodes coloured by agent |
| `run_<id>.png` | single runs drawn literally, failing runs first, up to 24 |

Pictures and pm4py models leave out messages and framework bookkeeping: assistant_message, create_agent, invoke_agent, invoke_workflow, llm_call, user_message, user_prompt. The logs keep everything.
