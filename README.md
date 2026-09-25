# IoT packet-2 device-disjoint measurement study

Reproducibility archive for the workshop paper **"Does a packet-2 IoT intrusion detector learn
the attack or the victim device?"** (submitted to SecuPriv-IoT 2026).

The paper is an honest measurement / negative-results study: it maps, per attack family, whether a
machine-learning intrusion detector that decides from a flow's first two packets generalizes to
**unseen victim devices**, using strictly device-disjoint evaluation across three public IoT
datasets (CICIoT2023, CICIoMT2024, CIC IIoT 2025).

## Layout

```
experiments/ciciot2023_census/ analysis scripts (*.py) + result JSONs (*.json)
```

The paper itself is not part of this repository.

- `experiments/.../paper_numbers.json` is the source of truth: every number in the paper is
  consolidated here by `consolidate_paper_numbers.py`. Each result JSON is the output of the
  script of the same theme (e.g. `feature_ablation_iiot2025.py` → `feature_ablation_iiot2025.json`).

## Data is NOT included (and never will be)

The raw packet captures and per-window CSVs are **license-bound public datasets** and are not
redistributed here. Obtain them from their sources:

- **CICIoT2023** — Canadian Institute for Cybersecurity, University of New Brunswick.
- **CICIoMT2024** — Canadian Institute for Cybersecurity, UNB.
- **CIC IIoT 2025 (DataSense)** — Canadian Institute for Cybersecurity, UNB.

The scripts read datasets from `./data/` and write intermediate caches to `./cache/` (both relative to
the repo root; create them and drop your dataset copies in). Place the CICIoT2023 archive at
`./data/CICIOT23/archive.zip`, the CIC IIoT 2025 extract under `./data/iiot/extract`, and the DataSense
windowed CSV at `./data/iiot/combined_dataset.csv`. Several scripts also accept env-var overrides
(`CICIOT_ZIP`, `IIOT2025_CSV`, `TSHARK_CAP`, `CACHE_DIR`, …), shown at the top of each file. Device
identity (MAC/IP) is used **only** to build evaluation splits — never as a model feature.

## Reproducing a result

```bash
python -m venv venv && ./venv/bin/pip install numpy pandas scikit-learn xgboost
# point the script at your dataset copy, then e.g.:
CICIOT_ZIP=/path/to/CICIOT23/archive.zip ./venv/bin/python experiments/ciciot2023_census/feature_ablation.py
```

Determinism: XGBoost 3.3.0 (`hist`), seed 42, `n_jobs=1`; scikit-learn 1.9.0; numpy 2.4.4.

## Authors

Amro Baseet and İsmail Bütün, Sakarya University. Funded by TÜBİTAK (BIDEB-2232/A, Grant 121C083).

## Scope and honesty

Results are bounded to the CIC IoT dataset lineage and one model class; the early-decision mechanism
is prior art and is not claimed as a contribution. See the paper's Limitations section.
