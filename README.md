# public-domain legal evaluation with abstention

The runs behind the legal evaluation in [Bounding hallucinations: Merlin-Arthur
protocols for mutual information bounds in language
models](https://aleph-alpha.com/en/blog/bounding-hallucinations-merlin-arthur-protocols-for-mutual-information-bounds-in-language-models).
The post explains the task, the prompts, how the runs were made and what came out
of them. This repo contains the full model inputs and outputs.

- `items.jsonl`, 100 lines, one JSON object per item: the gold verdict and the
  full user prompt.
- `runs.jsonl`, 600 lines, one JSON object per item and system, joined to the
  items by `item_id`.
- `system_prompt.txt`, plain text: the system prompt, which every run shared.

All three files are UTF-8 and the case documents are German, so read them with
an explicit encoding if your platform does not default to it.

The item worked through in the post is `f2fa2b9bbc17e3c4`.

## The task

Each `user_prompt` is in German and carries its own instructions, in five
sections: `<Szenario>`, `<Aufgabe>`, `<Eingabespezifikation>`,
`<Ausgabespezifikation>` and `<Eingabe>`. The last section holds the
Tatbestandsmerkmal, the legal condition to be checked, followed by the excerpts
from the case documents. The prompts run from about 2,800 to 79,600 characters,
with a median of 22,000.

A verdict is `True` where the condition is fulfilled, `False` where it is not,
and `Reject` where the documents do not settle the question.

On 22 of the 100 items the case file does not settle the question, so the
correct verdict is inconclusive (`Reject`). Those are the items the post looks
at most closely. On the remaining 78 the file does settle it, 43 in favour of
the condition and 35 against.

## Where the data comes from

The documents are German legal and administrative files from a planning approval
procedure (Planfeststellungsverfahren) under the Energiewirtschaftsgesetz, for
the construction and operation of a hydrogen pipeline with a compressor station:
the project "H2-Netzanschluss Hanekenfähr" by Nowega GmbH, which feeds hydrogen
produced in the RWE hydrogen park in Lingen into an existing pipeline network.
The procedure was conducted by the Landesamt für Bergbau, Energie und Geologie
(LBEG), file reference L1.4/L67304_02_01/2023-0002.

The planning approval decision and the approved documents were put on public
display from 10 October 2023 to 23 October 2023, electronically under
§ 3 Abs. 1 PlanSiG, and could additionally be inspected by arrangement at the
Stadt Lingen during office hours. The announcement is here:

<https://www.lingen.de/politik-rathaus-service/veroeffentlichungen/bekanntmachungen/planfeststellungsverfahren-fuer-die-errichtung-und-den-betri.html>

We removed PII from the data before storing and further processing.

## Fields

`items.jsonl`:

| Field | Meaning |
| --- | --- |
| `item_id` | Stable identifier of the item, shared by all six systems |
| `target` | Gold verdict: `True`, `False`, or `Reject` where the file does not settle it |
| `user_prompt` | The full user prompt, as described under "The task" |

`runs.jsonl`:

| Field | Meaning |
| --- | --- |
| `item_id` | The item this run answered, joins to `items.jsonl` |
| `model` | Short identifier of the system |
| `model_label` | The name used for the system in the blog post |
| `model_spec` | Provider slug, or a description of the checkpoint for the internal models |
| `run_date` | 2026-08-17 for the two internal models, 2026-08-12 for the four others |
| `reasoning_effort` | `high` for the other systems, null for the internal ones |
| `reasoning` | Reasoning text where the provider returned any |
| `answer` | The answer text |
| `raw_response` | The unparsed response the run produced |
| `prediction` | The verdict parsed out of the answer, null where none could be parsed |
| `correct` | Whether `prediction` equals `target` |
| `finish_reason` | `stop`, or `length` where the system ran into its token limit |
| `prompt_tokens`, `completion_tokens` | Token counts reported for the run |
| `reasoning_tokens` | As reported by the API, null for the two internal models |

Frontier models can change behind a fixed name, so a rerun through OpenRouter today
will not necessarily reproduce these answers, therefore we report `run_date` above.


## Loading

```python
import pandas as pd

system_prompt = open("system_prompt.txt", encoding="utf-8").read()
items = pd.read_json("items.jsonl", lines=True)
runs = pd.read_json("runs.jsonl", lines=True)
df = runs.merge(items, on="item_id")

df.groupby("model_label").correct.sum()

df[df.item_id == "f2fa2b9bbc17e3c4"]
```
