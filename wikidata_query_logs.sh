#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --gres gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -a 1-20%1
#SBATCH -J wikidata-query-logs
#SBATCH -e %x_%A_%a.err
#SBATCH -o %x_%A_%a.out

skip=${SKIP:-0}
take=${TAKE:-1000000}

srun python scripts/generate_queries_from_sparql.py \
  data/wikidata-query-logs/organic.raw.jsonl \
  data/wikidata-query-logs/organic-qwen25-72b \
  -kg wikidata \
  --kg-endpoint https://qlever.cs.uni-freiburg.de/api/wikidata-sebastian \
  --model Qwen/Qwen2.5-72B-Instruct \
  --skip "$skip" \
  --take "$take" \
  -b 8
