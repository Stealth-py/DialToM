# DialToM

Official code and data repository for the paper "*DialToM: A Theory of Mind Benchmark for Forecasting State-Driven Dialogue Trajectories*", submitted to KDD 2026 DnB track.

---

## Directory Structure

- `counterfactual_data`: contains three files, for each dataset, containing all counterfactuals generated for the counterfactual ablation study.
- `original_data`: contains the originally curated data for all three datasets.
- `verified_data`: contains the human verified further filtered version of `original_data`.

---

## Packages required

The user needs to install the following packages in their preferred virtual environment: `google-genai`, `openai`, `sacrebleu`, `rouge`, `bert-score`.

---

## Usage

### Benchmarking

#### Retrospective

> `python benchmark.py --model gpt-5 --task retrospective --filename retrospective.csv`

#### Prospective

> `python benchmark.py --model gpt-5 --task prospective --exp [EXP_TYPE] --filename prospective.csv`

- The user can choose between four experiment types for the prospective task (EXP_TYPE): `normal`, `easy`, `NOTA`, `CoT`. The filename will change dynamically based on what experiment is chosen to `{filename}_{EXP_TYPE}.csv`.
- `normal` experiment is the default prospective baseline, `easy` refers to the easy set evaluation of the prospective task, `NOTA` and `CoT` are the two ablations as discussed in the rebuttal phase.

#### Written

> > `python benchmark.py --model gpt-5 --task written --filename written.csv`


### Counterfactual testing

> `python counterfactual_test.py --model gpt-5 --filename counter.csv`


### Memorization pilot study

> `python memorization_pilot.py --model gpt-5 --filename memorize.csv`

---