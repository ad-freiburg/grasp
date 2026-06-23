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
