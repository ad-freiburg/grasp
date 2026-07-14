# GRISP: Guided Recurrent IRI Selection over SPARQL Skeletons

GRISP is a question-answering method for arbitrary knowledge graphs
that works by fine-tuning and running a small large language model
in a generate-then-retrieve fashion. The retrieval is performed
iteratively using search and list-wise re-ranking, and guided by
the model's own generated SPARQL skeletons and the knowledge graph structure.

You can also [download](https://ad-publications.cs.uni-freiburg.de/grisp/)
and run our fine-tuned models for Freebase and Wikidata.

## Installation

See the [main README](../../../../README.md) for installation instructions
of GRASP. Then make sure you have the GRISP extension installed as well:

```bash
pip install -e ".[grisp]"
```

> Note: All commands shown below have useful additional options that
> are not shown here for simplicity. Please check the help messages
> of the respective commands for more details.

## Train a GRISP model

An exemplary training config with the most important training options
is provided [`here`](../../../../configs/grisp/train.yaml).

We will show a running example with the WikiWebQuestions dataset for
Wikidata. Make sure that you have set up the corresponding GRASP
indices for the knowledge graph you want to work with before
starting with the steps below. Again, see the [main README](../../../../README.md)
for instructions on how to do this.

1: Prepare your data in jsonl format

```bash
# see an example of the data format
cat data/benchmark/wikidata/wwq/train.jsonl | head -1 | jq
```

2: Convert the data to the GRISP format

```bash
# convert training data
python -m grasp.baselines.grisp.data \
  wikidata \
  data/benchmark/wikidata/wwq/train.jsonl \
  data/grisp/wikidata/wwq/train.jsonl

# same for validation data
python -m grasp.baselines.grisp.data \
  wikidata \
  data/benchmark/wikidata/wwq/val.jsonl \
  data/grisp/wikidata/wwq/val.jsonl
```

3: Materialize the GRISP training data (Optional but recommended)

GRISP performs online data preprocessing and augmentation during training,
which is computationally expensive. However, you can also do this offline
and generate samples for a given number of epochs in advance. This is
recommended because it requires you to do it only once, and speeds up
training significantly.

```bash
# materialize training data for 4 epochs
python -m grasp.baselines.grisp.materialize \
  data/grisp/wikidata/wwq/train.jsonl \
  data/grisp/wikidata/wwq/train.materialized.jsonl \
  4 # number of epochs to materialize data for

# materialize validation data, which should stay the same across
# epochs, so we only need to do it for 1 epoch
# note: the --is-val flag is important to avoid data
# augmentations for validation data
python -m grasp.baselines.grisp.materialize \
  data/grisp/wikidata/wwq/val.jsonl \
  data/grisp/wikidata/wwq/val.materialized.jsonl \
  1 \
  --is-val
```

4: Train the model

Adapt the training config to your setup, then run the training command.
If you have materialized the data, you should set the `materialized`
option to `true` and point the train and validation data paths
to the materialized data.

```bash
python -m grasp.baselines.grisp.train \
  configs/grisp/train.yaml \
  data/grisp/runs/my-wikidata-wwq-model # output directory
```

> Note: We use Wandb for experiment tracking by default, so
> set WANDB_PROJECT and WANDB_ENTITY environment variables
> to log your runs to your Wandb account.

### Bootstrap query validation and skeleton improvement (optional)

Besides the two core tasks (skeleton generation and listwise IRI selection),
GRISP can additionally be trained on two bootstrapped tasks:

3. **Query validation** — a binary classifier deciding whether a final query
   correctly answers the question. It extends the simple empty-result check.
4. **Skeleton improvement** — rewriting a wrong query skeleton into a corrected
   one, used to recover from the malformed skeletons that cause most failures.

These tasks are *bootstrapped* from a model already trained on tasks 1 and 2:
the whole procedure is run `num_bootstraps` times per question (each run samples
one skeleton and resolves it), yielding a spread of query qualities (with
`constrain` and `check_empty` enabled, hard questions also yield partial
skeletons that do not fully resolve). Each fully-resolved prediction is scored
by F1 against the gold query, and the result is turned into training data. The
validation target is a soft distribution `[f1, 1 - f1]` over `[valid, invalid]`;
the improvement target is the gold NL skeleton, and the input is the model's own
(bad) skeleton together with the query resolved from it -- its selections and
result preview (or the partially resolved query and an unavailable result, when
it did not fully resolve).

```bash
# 1) train a v1 model on tasks 1 + 2 (type 'both', as above)
# 2) mine bootstrapped data from it (run the procedure 4x per question)
python -m grasp.baselines.grisp.bootstrap \
  configs/grisp/run.yaml \
  data/grisp/runs/my-wikidata-wwq-model \
  data/grisp/wikidata/wwq/train.jsonl \
  data/grisp/wikidata/wwq/train.bootstrap.jsonl \
  4

# 3) materialize the gold data as before, then train v2 on all four tasks by
#    setting TYPE=all and listing both the gold and bootstrapped materialized
#    files in train_files (the bootstrap file is already materialized).
TYPE=all python -m grasp.baselines.grisp.train \
  configs/grisp/train.yaml \
  data/grisp/runs/my-wikidata-wwq-model-all
```

At inference time, enable the learned validation/improvement self-correction
loop via the run config (`improve_skeletons`, see
[`run.yaml`](../../../../configs/grisp/run.yaml)). The loop validates each
candidate query and asks the model to rewrite the skeleton before retrying
selection, keeping the highest-scoring candidate. `improve_threshold` controls
early-exit: when set, the first candidate clearing it is accepted immediately;
set it to `null` to always run every skeleton through all improvement rounds and
return the best-scoring candidate.

## Run a GRISP model

An exemplary run config with the most important inference options
is provided [`here`](../../../../configs/grisp/run.yaml).

### Use a pre-trained model

Pre-trained GRISP models for Wikidata and Freebase are available
[here](https://ad-publications.cs.uni-freiburg.de/grisp/). If you want to use
them, make sure to have set up the GRASP indices for the
corresponding knowledge graph before running the model.

For example, to download and use our Wikidata WDQL model based on Qwen2.5 7B:

```bash
# Download the model
wget https://ad-publications.cs.uni-freiburg.de/grisp/qwen-2.5-7b-instruct-lora-wikidata-wdql-both-05-02-26.tar.gz
# Extract it
tar -xzf qwen-2.5-7b-instruct-lora-wikidata-wdql-both-05-02-26.tar.gz
# Run on a single question
python -m grasp.baselines.grisp.run \
  configs/grisp/run.yaml \
  qwen-2.5-7b-instruct-lora-wikidata-wdql-both-05-02-26 \
  run \
  --input "Where was Angela Merkel born?"

# Or run on a file with questions, e.g. the WWQ test set
python -m grasp.baselines.grisp.run \
  configs/grisp/run.yaml \
  qwen-2.5-7b-instruct-lora-wikidata-wdql-both-05-02-26 \
  file \
  --input-file data/benchmark/wikidata/wwq/test.jsonl
```

### Run a custom model

If you trained your own model, just use your training output directory as
the model path.

```bash
python -m grasp.baselines.grisp.run \
  configs/grisp/run.yaml \
  data/grisp/runs/my-wikidata-wwq-model \ # Path to your training output directory
  file \
  --input-file data/benchmark/wikidata/wwq/test.jsonl
```

### IRI fill order

GRISP resolves a skeleton's natural-language placeholders one at a time. The
order in which they are filled is controlled by the `fill_order` run config
option (or the `FILL_ORDER` environment variable):

| `fill_order` | Description |
|---|---|
| `left-to-right` (default) | placeholders in document order |
| `right-to-left` | placeholders in reverse document order |
| `entities-then-properties` | all entity placeholders first, then properties, each in document order |
| `properties-then-entities` | all property placeholders first, then entities, each in document order |
| `triple-wise-entities-then-properties` | triple by triple (document order); within each triple entities first, then properties (so each property is filled after both its endpoints) |
| `random` | a fixed random permutation per skeleton |

The model is trained to be **order-agnostic** (during data generation a random
subset of placeholders is resolved in random order, see
`grisp.data.prepare_selection`), so a single model supports all these orders and
you can switch between them at inference without retraining. This makes
`fill_order` a convenient ablation knob.

Orders other than `left-to-right` let the constraint filter (`constrain`) see
already-resolved neighbours on *both* sides of the current placeholder, rather
than only to the left. Note that when non-left-to-right orders leave unresolved
placeholders to the left of the current one, those triples cannot contribute to
the constraint query and selection gracefully falls back to position-only
filtering for that step.

```bash
# e.g. resolve entities before properties
FILL_ORDER=entities-then-properties python -m grasp.baselines.grisp.run \
  configs/grisp/run.yaml \
  data/grisp/runs/my-wikidata-wwq-model \
  file \
  --input-file data/benchmark/wikidata/wwq/test.jsonl
```

## Serve a GRISP model

```bash
python -m grasp.baselines.grisp.server configs/grisp/serve.yaml
```

The server runs on port 6790 by default and exposes the following HTTP endpoints.

| Endpoint | Method | Description |
|---|---|---|
| `/knowledge_graphs` | GET | Returns list with the configured KG name |
| `/config` | GET | Returns the server configuration |
| `/run` | POST | Run GRISP on a single question |

#### `POST /run`

**Request body:**

```json
{"question": "Where was Angela Merkel born?"}
```

**Response:** GRISP output as a JSON object

**Error codes:** `503` server busy, `504` generation timeout

## Evaluate a GRISP model

The `grisp.run` command produces outputs that are compatible
with the ones of GRASP. So you can just use the `evaluate`
subcommand of GRASP to evaluate.

```bash
# to evaluate the test predictions with f1 score
grasp evaluate f1 wikidata \
  data/benchmark/wikidata/wwq/test.jsonl \
  data/grisp/runs/my-wikidata-wwq-model/predictions.jsonl
```

## Citation

If you use GRISP, please cite our paper:

```bibtex
@article{DBLP:journals/corr/abs-2604-21133,
  author       = {Sebastian Walter and
                  Hannah Bast},
  title        = {{GRISP:} Guided Recurrent {IRI} Selection over {SPARQL} Skeletons},
  journal      = {CoRR},
  volume       = {abs/2604.21133},
  year         = {2026}
}
```
