# WorkflowEvals

This code allows you to reproduce the results from our [evals site](https://evals.typesafe.ai).

The eval assumes the workflow and its code is correct and evaluates models against the largest models from the most well-known labs.

## Run

Install Python 3.13+ and [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync --locked
export TYPESAFE_API_KEY=your-key
uv run python run.py invoice_processing --limit 5
```

The default model is `typesafe:jev-1.13.0`. Override it with `--model provider:model`
using the provider's exact model ID, such as `--model openai:gpt-6-astra`. Supported
providers: `openai`, `anthropic`, `fireworks`, `groq`, `cerebras`, and `typesafe`.
For example, `fireworks:accounts/fireworks/models/glm-5p3`. Set the matching
provider's `*_API_KEY` environment variable. TypeSafe's model is
`typesafe:jev-1.13.0`; it uses the public API endpoint unless `--base-url` overrides it.

`core/pricing.json` uses the same exact `provider:model` keys. Missing rates produce
unknown cost.

Runs are saved in the repo's gitignored `runs/<workflow>/<name>/` folder. The name
defaults to the model name, using the last component for model IDs containing `/`.
The example above saves to `runs/invoice_processing/jev-1.13.0/`.
Use `--name invoices-v2` to choose another name.

Add `--dry-run` to check inputs without model calls. Repeat a command to resume
saved cases. Use `--thinking high` with LLM providers to request reasoning effort and `--workers N`
to set concurrency. `uv run python run.py --help` lists all options.

## Workflows

| Folder | Cases | Task |
| --- | ---: | --- |
| `evals/invoice_processing/` | 150 | Invoice approval and fraud triage under three policies |
| `evals/customer_service/` | 204 | Customer-service routing and actions under three policies |
| `evals/agent_trace_observability/` | 111 | Agent-run action selection under two review profiles |
| `evals/security_incidents/` | 240 | Security-alert triage and containment |

`evals/` contains one folder per workflow. `core/` contains shared data loading,
model clients, execution, and scoring.

Inputs and reference labels download automatically from the
[WorkflowEvals collection](https://huggingface.co/collections/typesafe/workflowevals)
and stay in the Hugging Face cache. Runs load the latest `main` revision by default;
`--dataset-revision` optionally selects another commit, tag, or branch.
The resolved revision is recorded in each run.

For private datasets, run `uv run hf auth login` or set `HF_TOKEN`. Public datasets
need no login. After downloading, `HF_HUB_OFFLINE=1` runs from the cache.

## Results

The final summary prints average estimated cost and elapsed time per selected case,
including resumed cases. An average is `unknown` if any case lacks that measurement.

The output directory contains `results.json` with decisions and model responses,
`scores.json` with metrics, and case checkpoints for resuming. Scores use the
downloaded OpenAI, Anthropic, and consensus references. Failed cases remain in the results; use `--retry-errors`
to rerun them. Changing the model, settings, data, or code requires a different run name.

All workflows include references. To also compare against your own model run,
add it with `--reference`:

```bash
uv run python run.py security_incidents --model openai:gpt-6-astra \
  --name security-reference
uv run python run.py security_incidents --model typesafe:jev-1.13.0 \
  --reference runs/security_incidents/security-reference --name security-typesafe
```

`--reference` works for every workflow. Scores measure agreement with the
reference model; a model is excluded from scoring against its own answers.

Metrics are `exact_actions` (matching action sets, including arguments) and
`primary_action` (matching primary actions), where applicable to the workflow.

## Plot

Export all saved runs for a workflow, using provider colors:

```bash
uv run --extra plot python plot.py invoice_processing
```

This writes `runs/invoice_processing/comparison.svg`, with one point per run,
model labels, and provider colors. Agreement is on the y axis; mean cost per case
is on a logarithmic x axis. Use `--metric exact|primary` and `--x-axis cost|time`
to choose the axes; the defaults are `exact` and `cost`.
Primary agreement requires primary actions.

Use `--out comparison.png` or `.pdf` for another format, and
`--names jev-1.13.0 invoices-v2` to select runs. `--reference` defaults to consensus;
`--policy` selects a policy. Runs must use matching inputs, policies, and reference
labels; scores are recalculated on their shared cases. Runs without a known,
positive cost or time for the selected x axis are reported and omitted.

Add `--hf` to include the published model runs from the latest Hugging Face dataset:

```bash
uv run --extra plot python plot.py invoice_processing --hf
```

This also works without local runs. `--names` selects local runs while `--hf` adds
the published runs.
The resolved dataset revision is printed. Downloads stay in the Hugging Face cache.
Scores are recomputed from the published decisions, and cost and timing use the
recorded per-case values.

Use `average` for an equal-weight average across all workflows:

```bash
uv run --extra plot python plot.py average --hf
```

This writes `runs/average/comparison.svg`. Each workflow contributes equally to
both agreement and mean cost or time, regardless of its case count. Runs match by
model, reasoning setting, and question mode. Repeated runs are averaged within
each workflow first; models missing any workflow are reported and omitted.
The same options work for local runs, including `--names` and `--x-axis time`.
An average requires the metric in every workflow, so use `exact`; customer service
has no primary actions. `--policy` applies only to individual workflows.

## License

The code is licensed under [Apache-2.0](LICENSE). Datasets are licensed separately
as specified in their Hugging Face repositories.
