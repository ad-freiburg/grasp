# GRASP - Generic Reasoning and SPARQL Generation across Knowledge Graphs

## News

- August 21st 2026:
  - GRASP won the [WikiKGQA 2026 challenge](https://wikikgqa.org) at ISWC 2026;
    our [paper](https://ad-publications.cs.uni-freiburg.de/WIKIKGQA_grasp_grisp_WB_2026.pdf) also evaluates GRISP
  - Results at [grasp.cs.uni-freiburg.de/wiki-kgqa-evaluation](https://grasp.cs.uni-freiburg.de/wiki-kgqa-evaluation)

- August 21st 2026:
  - Entity linking [paper](https://ad-publications.cs.uni-freiburg.de/ISWC_entity_linking_grasp_BWB_2026.pdf) accepted as a poster at [ISWC 2026](https://iswc2026.semanticweb.org/)
  - Try it at [grasp.cs.uni-freiburg.de/entity-linking](https://grasp.cs.uni-freiburg.de/entity-linking)

- April 22nd 2026:
  - arXiv preprint of GRISP (Guided Recurrent IRI Selection over SPARQL Skeletons),
    an alternative QA baseline included in this repository, available at
    [arxiv.org/abs/2604.21133](https://arxiv.org/abs/2604.21133)

<details>
<summary>Older news</summary>

- February 16th 2026:
  - GRASP has been used to generate WDQL, a dataset of Wikidata query logs
  - arXiv preprint of "The Wikidata Query Logs Dataset" available at
    [arxiv.org/abs/2602.14594](https://arxiv.org/abs/2602.14594)
  - WDQL has been accepted to [SIGIR 2026](https://sigir2026.org/)
  - Dataset available at
    [github.com/ad-freiburg/wikidata-query-logs](https://github.com/ad-freiburg/wikidata-query-logs)

- October 20th 2025:
  - GRASP can now be used for entity linking, in particular for
cell entity annotation
  - Workshop paper published for [SemTab 2025 challenge](https://sem-tab-challenge.github.io/2025/)
  - Preview of camera-ready version available [here](https://ad-publications.cs.uni-freiburg.de/SEMTAB_entity_linking_grasp_WB_2025.pdf)

- August 28th 2025:
  - Demo paper of GRASP has also been accepted to [ISWC 2025](https://iswc2025.semanticweb.org/)
  - Preview of camera-ready version available [here](https://ad-publications.cs.uni-freiburg.de/ISWC_grasp_demo_WB_2025.pdf)

- July 31st 2025:
  - GRASP has been accepted to [ISWC 2025](https://iswc2025.semanticweb.org/)
  - Preview of camera-ready version available [here](https://ad-publications.cs.uni-freiburg.de/ISWC_grasp_WB_2025.pdf)

- July 14th 2025:
  - arXiv preprint available at [arxiv.org/abs/2507.08107](https://arxiv.org/abs/2507.08107)

- July 10th 2025:
  - Code release
  - Data release

</details>

## Overview and directory structure

Links:

- Public demo available at [grasp.cs.uni-freiburg.de](https://grasp.cs.uni-freiburg.de)
- Data available at [ad-publications.cs.uni-freiburg.de/grasp](https://ad-publications.cs.uni-freiburg.de/grasp)

```
apps/
  evaluation/                     # Streamlit app for evaluation
  grasp/                          # Svelte web app compatible with GRASP server
  grisp/                          # Svelte web app compatible with GRISP server
bash/                             # Bash scripts to run and evaluate GRASP
configs/
  run.yaml                        # Config to run GRASP with a single KG
  serve.yaml                      # Config to run GRASP with all available KGs
  grisp/                          # Configs for the GRISP baseline
  notes/                          # Configs for note-taking
queries/                          # Custom index data and info SPARQL queries
                                    for various knowledge graphs
scripts/                          # Various helper scripts
data/                          
  benchmark/                      # Benchmarks grouped by knowledge graph
    [knowledge-graph]/
      [benchmark]/                   
        test.jsonl                # Test set with input and ground truth
        train-example-index/      # Index based on train set for few-shot learning
                                    (needs to be downloaded)
        outputs/
          [model].jsonl           # Model output
          [model].config.json     # Model config
          [model].evaluation.json # Evaluation against ground truth
  kg-index/                       # KG indices (need to be downloaded)
    wikidata/
    freebase/
    ...
src/                              # Source code for GRASP
Makefile                          # Makefile for building benchmarks
```

## Quickstart

Follow these steps to run GRASP. If you want to use Docker, see section
[Run GRASP with Docker](#run-grasp-with-docker) below.

### Run GRASP

1. Install GRASP

```bash
# Via git (recommended, up-to-date version)
pip install git+https://github.com/ad-freiburg/grasp.git@main

# From PyPI (not recommended as of now, at least as long
# as GRASP is under heavy development)
pip install grasp-rdf

# Optional extras:
# - evaluation: Streamlit apps (apps/evaluation, grasp evaluate expert)
# - grisp: training and running GRISP
# - data: rebuilding the benchmark datasets with the Makefile
pip install "grasp-rdf[evaluation,grisp,data] @ git+https://github.com/ad-freiburg/grasp.git@main"

# To install with the dependency versions we tested, use the constraints file
# from a clone of this repository:
pip install -c constraints.txt -e .
```

2. Set the `GRASP_INDEX_DIR` env variable. Defaults to `$HOME/.grasp/index` if not
set. We set it to `$PWD/data/kg-index`, but you can choose any directory you like.

3. Get indices for the knowledge graphs you want to use. All indices are available
[publicly](https://ad-publications.cs.uni-freiburg.de/grasp/kg-index).
For example, to get the indices for Wikidata:

```bash
# Change to index directory
cd $GRASP_INDEX_DIR
# Download Wikidata index
wget https://ad-publications.cs.uni-freiburg.de/grasp/kg-index/wikidata.tar.gz
# Extract index
tar -xzf wikidata.tar.gz
```

Optionally, you can also download example indices for few-shot learning.
Example indices are always built from the train set of a benchmark
and called `train-example-index`.
For example, to get the example index for QALD-10 on Wikidata:

```bash
# Change to benchmark directory
cd data/benchmark/wikidata/qald10
# Download example index
wget https://ad-publications.cs.uni-freiburg.de/grasp/benchmark/wikidata/qald10/train-example-index.tar.gz
# Extract example index
tar -xzf train-example-index.tar.gz
```

4. Run GRASP:

```bash
# Note, that if you e.g. run OpenAI models, you also need to set the
# OPENAI_API_KEY env variable (see section about supported models below).

# Tip: Set --log-level DEBUG to show the individual steps of GRASP
# (reasoning and function calls) in a nicely formatted way.

# Run GRASP on an input and output the result to stdout as JSON with metadata.
# Actual output for the task is in the "output" field of that JSON object.

# Input from stdin:
echo "Where was Angela Merkel born?" | grasp run configs/run.yaml

# Input via CLI argument:
grasp run configs/run.yaml --input "Where was Angela Merkel born?"

# You can run different tasks with GRASP (default is sparql-qa).
# Depending on the task, the expected input format and output format
# will differ. For general-qa, the input is also a natural language
# question, same as for sparql-qa, but the output will be just a natural
# language answer instead of a SPARQL query.
echo "Where was Angela Merkel born?" | grasp run configs/run.yaml --task general-qa

# For cell entity annotation (cea), the input is a JSON object with a "table"
# field containing "header" and "data". The task links table cells to entities
# in the knowledge graph.
grasp run configs/run.yaml --task cea --input-format json \
  --input '{"table": {"header": ["Country", "Capital"], "data": [["France", "Paris"]]}}'

# For entity linking (entity-linking), the input is a JSON object with a "text"
# field containing "data" (the text to annotate) and optionally
# "annotate_start"/"annotate_end" (character offsets restricting the span to
# annotate) and "special_instructions". The task links mentions in the text to
# entities in the knowledge graph.
grasp run configs/run.yaml --task entity-linking --input-format json \
  --input '{"text": {"data": "Angela Merkel was born in Hamburg."}}'

# For SPARQL-to-question (sparql-to-question), the input is a SPARQL query and
# the output are natural language questions that the query answers.
grasp run configs/run.yaml --task sparql-to-question \
  --input 'SELECT ?x WHERE { wd:Q567 wdt:P19 ?x }'

# Show all available options:
grasp run -h

# You can also run GRASP on multiple inputs (in JSONL format).
# In the following, we show an example to run GRASP on the QALD-10 
# test set over Wikidata.

# Input from stdin:
cat data/benchmark/wikidata/qald10/test.jsonl | grasp file configs/run.yaml

# Input via CLI argument:
grasp file configs/run.yaml --input-file data/benchmark/wikidata/qald10/test.jsonl

# Save output to a file instead of stdout and show progress bar:
grasp file configs/run.yaml \
  --input-file data/benchmark/wikidata/qald10/test.jsonl \
  --output-file data/benchmark/wikidata/qald10/outputs/test.jsonl \
  --progress

# Show all available options:
grasp file -h

# Visualize the interaction trace from a GRASP output file (reads JSONL from stdin):
cat data/benchmark/wikidata/qald10/outputs/gpt-41.jsonl | grasp show
# Skip system/config/function messages for a more compact view:
cat data/benchmark/wikidata/qald10/outputs/gpt-41.jsonl | grasp show --skip-system

# You can also run GRASP in a client-server setup. This is also the server
# that powers the corresponding web app.
# To start a GRASP server, by default on port 6789, just run:
grasp serve configs/run.yaml

# For convenience, we also provide a config to run the server with all
# available knowledge graphs (make sure to download all indices first):
grasp serve configs/serve.yaml

# Show all available options:
grasp serve -h

# Evaluate GRASP output with F1 score (for sparql-qa task) by executing
# predicted and ground truth SPARQL queries and comparing results:
grasp evaluate f1 wikidata \
  data/benchmark/wikidata/qald10/test.jsonl \
  data/benchmark/wikidata/qald10/outputs/gpt-41.search_extended.jsonl

# Use a judge model to pick the best output from multiple prediction files.
# The last argument is the output evaluation file:
grasp evaluate judge configs/run.yaml \
  data/benchmark/wikidata/qald10/test.jsonl \
  data/benchmark/wikidata/qald10/outputs/model1.jsonl \
  data/benchmark/wikidata/qald10/outputs/model2.jsonl \
  data/benchmark/wikidata/qald10/outputs/judge.evaluation.json

# Launch a blind expert evaluation app (Streamlit) to manually compare outputs:
grasp evaluate expert \
  data/benchmark/wikidata/qald10/test.jsonl \
  data/benchmark/wikidata/qald10/outputs/model1.jsonl \
  data/benchmark/wikidata/qald10/outputs/model2.jsonl \
  data/benchmark/wikidata/qald10/outputs/expert.evaluation.json

# By default the app is served under /expert; override it if needed:
grasp evaluate expert --base-url-path /custom-expert \
  data/benchmark/wikidata/qald10/test.jsonl \
  data/benchmark/wikidata/qald10/outputs/model1.jsonl \
  data/benchmark/wikidata/qald10/outputs/model2.jsonl \
  data/benchmark/wikidata/qald10/outputs/expert.evaluation.json

# Show all available options:
grasp evaluate f1 -h
grasp evaluate judge -h
grasp evaluate expert -h

# Build an example index for few-shot learning from a JSONL file of examples.
# The last argument is a short description of the examples, which is shown
# to the model in the system prompt:
grasp examples data/benchmark/wikidata/qald10/train.jsonl \
  data/benchmark/wikidata/qald10/train-example-index \
  "Question-SPARQL pairs from the QALD-10 train set over Wikidata"

# Merge data from multiple knowledge graphs into a single combined KG.
# The first KG is the primary one; entities/properties from subsequent KGs
# are added to it. For example used to combine language-specific indices
# of the same knowledge graph:
grasp merge wikidata-en wikidata-de wikidata-fr entities wikidata-multilingual
grasp merge wikidata-en wikidata-de wikidata-fr properties wikidata-multilingual

# Note-taking: run GRASP on a knowledge graph to produce notes that
# can be included in the config to improve performance.
# Take notes by running GRASP on exemplary task samples:
grasp notes samples configs/notes/samples.yaml notes/
# Take notes from existing GRASP output files:
grasp notes outputs configs/notes/outputs.yaml notes/
# Take notes by freely exploring a knowledge graph (no task samples needed):
grasp notes explore configs/notes/explore.yaml notes/
# Generate plausible questions over the configured KGs as bootstrap samples
# for note taking (no task samples needed):
grasp notes generate-questions configs/notes/generate_questions.yaml notes/
```

### Server API

When running `grasp serve`, the server exposes the following HTTP endpoints (by default on port 6789).

| Endpoint | Method | Description |
|---|---|---|
| `/knowledge_graphs` | GET | Returns list of available KG names |
| `/config` | GET | Returns the server configuration |
| `/run` | POST | Run GRASP on a single input |
| `/live` | WebSocket | Run GRASP and stream its steps (used by the web app) |
| `/save` | POST | Save a web app state for sharing (only if `share` is set) |
| `/load/{share_id}` | GET | Load a shared web app state (only if `share` is set) |
| `/transcribe` | POST | Transcribe an audio question (only if `speech_to_text` is set) |

Server-specific config options (see `ServerConfig` in the
[config code](src/grasp/configs.py)):

- `port`, `max_connections`, `max_time` / `grace_time`, `max_idle_time`
- `log_outputs` / `log_file`: log all outputs / server logs to these files
- `share`: directory to store shared web app states in
- `rate_limit` / `rate_limit_window`: max requests per client IP per window (seconds)
- `speech_to_text`: model and endpoint for transcribing spoken questions

#### `POST /run`

**Request body:**

```json
{
  "task": "sparql-qa",
  "input": "Where was Angela Merkel born?",
  "knowledge_graphs": ["wikidata"],
  "past": null
}
```

- `task`: one of `"sparql-qa"`, `"general-qa"`, `"cea"`, `"entity-linking"`,
  `"sparql-to-question"`, `"wikidata-query-logs"`
- `input`: the task input (string for most tasks; for `"cea"` a JSON object with
  `header` and `data`, for `"entity-linking"` a JSON object with `data` and the
  optional fields shown above)
- `knowledge_graphs`: list of one or more KG names available on the server
- `past` *(optional)*: conversation history for multi-turn interactions, with fields `messages` (list of previous messages) and `known` (set of known entity/property IRIs)

**Response:** GRASP output as a JSON object

**Error codes:** `400` invalid KG selection, `429` rate limit exceeded, `500` internal error, `503` server busy, `504` generation timeout

### Run GRASP with Docker

Build the Docker image:

```bash
docker build -t grasp .
```

The image installs GRASP with the `evaluation` extra and the tested dependency
versions from `constraints.txt`. The entrypoint for the Docker image is the
`grasp` CLI. To run it with Docker, make sure that your `GRASP_INDEX_DIR` is
mounted to `/opt/grasp` and your API keys (e.g. `OPENAI_API_KEY`) are set as
env variables.

Some example commands are shown below.

```bash
# Answer a single question from stdin
echo "Where was Angela Merkel born?" | \
  docker run -i --rm \
  --user $(id -u):$(id -g) \
  -e OPENAI_API_KEY \
  -v $GRASP_INDEX_DIR:/opt/grasp \
  -e HF_HOME=/hf \
  -v $HF_HOME:/hf \
  grasp run configs/run.yaml

# If you want to run a server with your own config,
# just mount it into the container
docker run --rm \
  --user $(id -u):$(id -g) \
  -e OPENAI_API_KEY \
  -v $GRASP_INDEX_DIR:/opt/grasp \
  -e HF_HOME=/hf \
  -v $HF_HOME:/hf \
  -v $PWD/my_config.yaml:/grasp/server.yaml \
  grasp serve server.yaml
```

### Configure GRASP

GRASP can be configured via a single YAML config file, which is passed
to `grasp run`, `grasp file`, or `grasp serve` as first argument (see above).
You can use env variable placeholders in the config file of the form
`env(VAR_NAME:default_value)`, which will be replaced at runtime by the value of
the env variable `VAR_NAME` if it is set, or by `default_value` otherwise.
If no default value is given and the env variable is not set, an error
is raised. If you omit an entire config option, we also use a default value
as specified in the config code.

The configuration options and the use of env variable placeholders are
mostly self-explanatory, so we refer you to the [example config files](configs)
and the [config code](src/grasp/configs.py) for details. Some options that
are less obvious are described below.

**Knowledge graphs and indices**

Each entry in `knowledge_graphs` selects a KG and the indices to load for it:

```yaml
knowledge_graphs:
  - kg: wikidata
    entities: fuzzy        # auto, fuzzy, embedding, keyword, or null to disable
    properties: embedding
    literals: auto         # fuzzy for more than 1M literals, embedding otherwise
    shapes: {}             # shape index with default settings, null to disable
    examples: data/benchmark/wikidata/qald10/train-example-index
    notes_file: null
```

**SPARQL endpoint**

By default, GRASP queries the QLever endpoint at `https://qlever.dev/api/<kg_name>`.
To use another endpoint, set `endpoint` in the KG's `info.json` (see
[below](#customizing-prefixes-and-info-sparql-queries)) or override it per
KG in the config. Extra HTTP headers and query parameters for every request can
be set the same way:

```yaml
knowledge_graphs:
  - kg: my-kg
    info:
      endpoint: http://localhost:7001
      headers: {X-Api-Key: ...}
      params: {timeout: 30s}
```

Fields set under `info` in the config overwrite the ones from `info.json`.

GRASP sends its query timeout to QLever endpoints, so QLever stops queries
GRASP has given up on. For other SPARQL servers, set `backend: other` under
`info`.

**Basic Auth**

If an endpoint requires Basic Auth, set the env variables
`GRASP_BASICAUTH_<KG>_USER` and `GRASP_BASICAUTH_<KG>_PASSWD`, where `<KG>` is
the KG name in upper case with non-alphanumeric characters replaced by `_`
(e.g. `GRASP_BASICAUTH_MY_KG_USER` for `my-kg`).

**Search parameters**

Embedding indices store their search parameters (`min_score`, `rerank`,
`exact`) with the index (see `grasp index -h`). You can override them per index
in the config:

```yaml
knowledge_graphs:
  - kg: wikidata
    properties:
      type: embedding
      params: {min_score: 0.5}
```

**Additional indices**

Besides entities, properties, and literals, you can load further indices built
with [search-rdf](https://github.com/ad-freiburg/search-rdf). Put the search-rdf
build config at `$GRASP_INDEX_DIR/<kg_name>/indices.yaml` and build the
indices into `$GRASP_INDEX_DIR/<kg_name>/<name>`. GRASP reads `name`, `type`,
and `description` of each entry under `indices`, and loads the ones you select
in the config via `indices: [<name>, ...]`.

**Task options**

Task-specific options are set under `task_kwargs`, keyed by task name, e.g.:

```yaml
task_kwargs:
  sparql-qa:
    know_before_answer: true   # only allow IRIs seen in earlier function call results
  cea:
    context: null              # rows of table context shown to the model
    know_before_annotate: true
    show_state_after_annotate: false  # show the rows to annotate after each (de-)annotation
  entity-linking:
    context: null              # characters of surrounding text shown to the model
    know_before_annotate: true
    show_state_after_annotate: true
  sparql-to-question:
    max_questions: 3
    language: English
```

See the respective task code in [src/grasp/tasks](src/grasp/tasks) for all
options and defaults.

### Build your own knowledge graph indices

Using GRASP with your own knowledge graph requires two steps:

- Getting the index data from a SPARQL endpoint for the knowledge graph
- Building the indices

You can do this manually (see below) or use `grasp setup`, which automatically
generates the necessary SPARQL queries and configuration by exploring the endpoint:

```bash
# Auto-configure a KG by exploring its SPARQL endpoint.
# 'standard' runs info + all three index phases (entities, properties, literals).
grasp setup wikidata standard configs/run.yaml

# Run only specific phases:
grasp setup wikidata info configs/run.yaml
grasp setup wikidata indices configs/run.yaml  # entities + properties + literals

# Show all available options:
grasp setup -h
```

#### Get index data

We get the index data by issuing SPARQL queries to a SPARQL endpoint.
GRASP supports three index types: **entities**, **properties**, and **literals**
(literal values such as names, labels, or other string-valued data that appear
as objects in the graph). Each query must return rows with the following columns:

1. `?id` — the IRI of the entity/property, or the literal value itself (required)
2. `?value` — a single label or alias (one row per label/alias)
3. `?tags` — optional comma-separated tags for the value, e.g., `"main"` to mark the primary label

Each item can appear in multiple rows (one per label or alias).
Rows for the same `?id` must be consecutive (i.e., results should be ordered
by `?id`).

A typical SPARQL query for that looks like this:

```sparql
SELECT DISTINCT ?id ?value ?tags WHERE {
  {
    # main label — tagged as "main" so it is used as the primary search field
    ?id rdfs:label ?value
    BIND("main" AS ?tags)
    FILTER(LANG(?value) = "en")
  } UNION {
    # aliases / alternative labels
    ?id skos:altLabel ?value
    FILTER(LANG(?value) = "en")
  }
}
ORDER BY ?id DESC(?tags)
```

The query body will determine which entities, properties, or literals are included
in the index, and how their labels and aliases are retrieved.

> Notes:
>
> - If you do not provide custom index data SPARQL queries, we use the generic
> default queries from [here](src/grasp/sparql/queries)
> - Our custom index data queries for various knowledge graphs
> are [here](queries)
> - By default (`--add-id-as-label auto`), entities without label or alias get
> no fallback, properties always get their (prefixed) IRI as additional label to
> make them searchable by IRI, and literals without label use their value as label.
> Change this with `--add-id-as-label <never|always|empty>`
> (`empty` means only for items without any label or alias)

With the CLI, you can use the `grasp data` command as follows:

```bash
# The index type (entities, properties, or literals) is a required argument.
# The SPARQL endpoint is taken from --endpoint, the endpoint in
# $GRASP_INDEX_DIR/<kg_name>/info.json, or https://qlever.dev/api/<kg_name>
# (in that order). Headers, query parameters, and Basic Auth are applied as
# described in section "Configure GRASP".
# The data will be saved to $GRASP_INDEX_DIR/<kg_name>/entities/data.jsonl
# and $GRASP_INDEX_DIR/<kg_name>/properties/data.jsonl etc.
# For example, to get the index data for IMDB:
grasp data imdb entities
grasp data imdb properties
grasp data imdb literals

# You can also set a custom SPARQL endpoint:
grasp data my-imdb entities --endpoint https://my-imdb-sparql-endpoint.com/sparql

# To download the index data, we use generic queries by default. You can also
# provide your own queries, which is recommended especially for larger knowledge
# graphs or those with an unusual schema.
grasp data imdb entities --index-sparql <path/to/entity.sparql>
grasp data imdb properties --index-sparql <path/to/property.sparql>

# Replace {key} placeholders in the query, and send extra query parameters
# or HTTP headers with the request:
grasp data imdb entities --index-sparql <path/to/entity.sparql> \
  --replace lang:en --query-parameters timeout:600s --query-headers X-Api-Key:...

# Skip the live query and build the data from SPARQL results in JSON format
# that you downloaded before:
grasp data imdb entities --data-file <path/to/results.json>

# Show all available options:
grasp data -h
```

#### Build indices

After getting the index data, you can build the indices for the knowledge graph.
You probably do not need to change any parameters here.

With the CLI, you can use the `grasp index` command as follows:

```bash
# The index type (entities, properties, literals, or shapes) is a required argument.
# The built index will be saved to $GRASP_INDEX_DIR/<kg_name>/<index>/<index_type>.
# For example, to build the indices for IMDB:
grasp index imdb entities
grasp index imdb properties
grasp index imdb literals

# You can also change the type of index that is built via --index-type
# (not applicable for shapes).
# By default (auto), we build a fuzzy index for entities, an embedding
# index for properties, and for literals a fuzzy index if there are more than
# 1M of them, an embedding index otherwise.
grasp index imdb entities --index-type <keyword|fuzzy|embedding>
grasp index imdb properties --index-type <keyword|fuzzy|embedding>

# Embedding indices use Qwen/Qwen3-Embedding-0.6B by default (--emb-model,
# --emb-device, --emb-batch-size). With --min-score-percentile, a minimum
# match score is derived from the similarity of random item pairs; with
# --min-score, --rerank, and --exact you set the search parameters stored
# with the index directly (they can be overridden in the config, see
# "Search parameters" above).
grasp index imdb properties --index-type embedding --min-score-percentile

# For the shape index, a config file is required (see "Build a shape index" below):
grasp index imdb shapes configs/run.yaml

# Show all available options:
grasp index -h
```

After this step is done, you can use the knowledge graph with GRASP by
including it in your config file (see above).

#### Build a shape index

A **shape index** gives GRASP schema context about the knowledge graph's class
structure — which properties a given class of entities tends to have, their
datatypes, and cardinalities. GRASP uses this as LLM context for more accurate
SPARQL generation via the `search_shape` and `get_shape` functions.

```bash
# Step 1: run the shapes-setup agent to configure the class-typing pattern
# (e.g. wdt:P31 for Wikidata, rdf:type for RDF-style graphs):
grasp setup my-kg shapes configs/run.yaml

# Step 2: build the shape index (config required):
grasp index my-kg shapes configs/run.yaml

# Limit the number of classes to profile (default 500):
grasp index my-kg shapes configs/run.yaml --max-classes 200
```

Shape profiling is enabled by default. Set `shapes: null` in a `knowledge_graphs`
entry of your config to disable it.

#### Customizing prefixes and info SPARQL queries

There are two more optional steps you can perform to customize the behavior
of GRASP related to your knowledge graph.

**KG info (description, prefixes, and endpoint)**

First, you can provide metadata about a knowledge graph via an
`info.json` file. This file can contain a description of the knowledge graph
(shown to the model in the system prompt) and prefix mappings used at build
time and runtime.
Create a file `$GRASP_INDEX_DIR/<kg_name>/info.json`
in the following format (example for Wikidata):

```jsonc
{
  "description": "A free, collaborative, multilingual knowledge base maintained by the Wikimedia Foundation.",
  "prefixes": {
    "wd": "http://www.wikidata.org/entity/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    // other prefixes ...
  },
  "endpoint": "https://qlever.dev/api/wikidata",
  "headers": {},
  "params": {}
}
```

All fields are optional. The description is shown to the model alongside the
knowledge graph name and endpoint. The prefixes are used during build time for
fallback label generation (see `--add-id-as-label` above). During runtime, the
prefixes are used to shorten IRIs in function call results, and allow GRASP to
use prefixed instead of full IRIs in function call arguments. Common prefixes
like `rdf`, `rdfs`, `owl`, `xsd`, or `skos` are always available.
`endpoint`, `headers`, and `params` set the SPARQL endpoint and extra HTTP
headers and query parameters used for this knowledge graph (default endpoint is
`https://qlever.dev/api/<kg_name>`); they can be overridden in the config (see
"SPARQL endpoint" in section "Configure GRASP").

> Note: Prefixes are not retrieved from the endpoint automatically. Run
> `grasp setup <kg_name> info <config>` to let an agent create `info.json`
> for you.

**Info SPARQL queries**

Second, you can customize the SPARQL queries that GRASP uses to fetch additional
information about entities and properties for enriching search results.
For that, create a file `$GRASP_INDEX_DIR/<kg_name>/entities/info.sparql`
for entities or `$GRASP_INDEX_DIR/<kg_name>/properties/info.sparql` for properties.
The file should contain a SPARQL query that returns three columns in its results:

1. `?id`: the IRI of the entity/property (required)
2. `?value`: a single piece of additional information (e.g. a label, alias, or description)
3. `?type`: the type of information, one of `"label"`, `"alias"`, or `"other"`

The query returns one row per piece of information (not one row per entity).
A typical SPARQL query for that looks like this:

```sparql
SELECT DISTINCT ?id ?value ?type WHERE {
  {
    VALUES ?id { {IDS} }
    ?id rdfs:label ?value
    BIND("label" AS ?type)
  } UNION {
    VALUES ?id { {IDS} }
    ?id skos:altLabel ?value
    BIND("alias" AS ?type)
  } UNION {
    VALUES ?id { {IDS} }
    ?id rdfs:comment ?value
    BIND("other" AS ?type)
  }
  FILTER(LANG(?value) = "en")
}
ORDER BY ?id ?type ?value
```

At runtime, all places where `{IDS}` appears in the query will be
replaced by the list of entity/property IRIs to get information for.
Typically, this will be within a `VALUES ?id { ... }` clause as
shown above.

See our [info SPARQL query for Wikidata entities](queries/wikidata.entity.info.sparql) as an example.

> Note: If no custom info SPARQL query is found, we use the
> default ones from [here](src/grasp/sparql/queries)

## Run GRASP webapp

Make sure to start a GRASP server first (see above).
Then follow [these instructions](apps/grasp/README.md) to run the GRASP web app.

## Run GRISP baseline

GRISP (Guided Recurrent IRI Selection over SPARQL Skeletons) is an alternative
question-answering baseline included in this repository. It works by fine-tuning
a small language model to generate SPARQL skeletons, then iteratively retrieving
and re-ranking entities using the GRASP search indices.

See the arXiv preprint at [arxiv.org/abs/2604.21133](https://arxiv.org/abs/2604.21133)
for details.

Follow [these instructions](src/grasp/baselines/grisp/README.md) to train,
run, and evaluate GRISP. To run the GRISP web app, follow
[these instructions](apps/grisp/README.md).

## Run evaluation app

Follow [these instructions](apps/evaluation/README.md) to run the
evaluation app for the SPARQL QA task.

## Supported models

GRASP supports OpenAI models (via the Responses or Chat Completions API),
Anthropic models (via the Messages API), and any OpenAI-compatible API endpoint
(e.g. vLLM, SGLang, Ollama, or Gemini). The model is set in the config file
(or via env variables, see [configs/run.yaml](configs/run.yaml)):

- `model`: the model name as expected by the API
- `model_provider`: `openai/responses` (default, Responses API),
  `openai/completions` (Chat Completions API, also for OpenAI-compatible
  endpoints), or `anthropic` (Messages API)
- `model_endpoint`: base URL of the API endpoint (defaults to the official
  endpoint of the provider)
- `model_api_key`: API key (if unset, the key is read from the env variable of
  the provider's SDK, i.e. `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)
- `model_kwargs`: additional inference parameters passed to the API as is
  (e.g. reasoning effort, or `extra_body` for vLLM-specific sampling parameters)

Note that the `model_kwargs` in [configs/run.yaml](configs/run.yaml)
(`reasoning`, `text`) are specific to the OpenAI Responses API. Replace them
when using another provider, otherwise the API call fails.

For example, to serve an open model with vLLM (adapt the tensor parallel size to
your GPU setup):

```bash
vllm serve Qwen/Qwen3.5-27B --reasoning-parser qwen3 \
  --tool-call-parser hermes --enable-auto-tool-choice --port 8000
```

and use it with GRASP via these settings in your config:

```yaml
model: Qwen/Qwen3.5-27B
model_provider: openai/completions
model_endpoint: http://localhost:8000/v1
model_api_key: dummy
model_kwargs:
  temperature: 0.6
  # vLLM-specific sampling parameters go into extra_body
  extra_body:
    top_k: 20
```

## Misc

To prepare some benchmark datasets with the [Makefile](Makefile),
e.g. using `make wikidata-benchmarks`, you need the `data` extra
(`pip install -e ".[data]"`) and first need to clone
[github.com/KGQA/KGQA-datasets](https://github.com/KGQA/KGQA-datasets) into `third_party`:

```bash
mkdir -p third_party
git clone https://github.com/KGQA/KGQA-datasets.git third_party/KGQA-datasets
```

## Citation

If you use this project, please consider citing the following works:

```bibtex
@inproceedings{DBLP:conf/semweb/WalterB25,
  author       = {Sebastian Walter and
                  Hannah Bast},
  title        = {{GRASP:} Generic Reasoning And {SPARQL} Generation Across Knowledge
                  Graphs},
  booktitle    = {{ISWC} {(1)}},
  series       = {Lecture Notes in Computer Science},
  volume       = {16140},
  pages        = {271--289},
  publisher    = {Springer},
  year         = {2025}
}

@inproceedings{DBLP:conf/semweb/WalterB25a,
  author       = {Sebastian Walter and
                  Hannah Bast},
  title        = {{GRASP:} Generic Reasoning And {SPARQL} Generation across Knowledge
                  Graphs - Demo System},
  booktitle    = {{ISWC} (Industry/Doctoral Consortium/Posters/Demos)},
  series       = {{CEUR} Workshop Proceedings},
  volume       = {4085},
  pages        = {475--482},
  publisher    = {CEUR-WS.org},
  year         = {2025}
}

@inproceedings{DBLP:conf/om2/0002B25,
  author       = {Sebastian Walter and
                  Hannah Bast},
  title        = {Knowledge Graph Entity Linking via Interactive Reasoning and Exploration
                  with {GRASP}},
  booktitle    = {OM@ISWC},
  series       = {{CEUR} Workshop Proceedings},
  volume       = {4144},
  pages        = {242--248},
  publisher    = {CEUR-WS.org},
  year         = {2025}
}

@inproceedings{DBLP:conf/sigir/WalterB26,
  author       = {Sebastian Walter and
                  Hannah Bast},
  title        = {The Wikidata Query Logs Dataset},
  booktitle    = {{SIGIR}},
  pages        = {3442--3447},
  publisher    = {{ACM}},
  year         = {2026}
}
```

The following papers have been accepted at ISWC 2026 and will appear soon:

- Marius Birmele, Sebastian Walter, Hannah Bast: Agentic Knowledge Graph Entity
  Linking. ISWC 2026 Posters and Demos
  ([PDF](https://ad-publications.cs.uni-freiburg.de/ISWC_entity_linking_grasp_BWB_2026.pdf))
- Sebastian Walter, Hannah Bast: GRASP and GRISP for Question Answering over
  Wikidata. WikiKGQA 2026 Challenge at ISWC 2026
  ([PDF](https://ad-publications.cs.uni-freiburg.de/WIKIKGQA_grasp_grisp_WB_2026.pdf))
