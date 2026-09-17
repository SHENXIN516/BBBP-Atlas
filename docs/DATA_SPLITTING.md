# Data splits

| Setting | Split seed | Train | Validation | Test | Files |
|---|---:|---:|---:|---:|---|
| Small molecules | 88 | 7,292 | 912 | 912 | [small_seed88](../splits/small_seed88/) |
| Mixed small molecules and peptides | 42 | 8,014 | 1,002 | 1,002 | [mixed_seed42](../splits/mixed_seed42/) |

Both settings use random 8:1:1 splitting without label stratification. An initial 80:20 split is followed by a 50:50 split of the remainder, using the same seed in both calls. Split seeds determine sample membership; model seeds determine training randomness.

Each directory contains `train.csv`, `val.csv`, `test.csv`, a complete `split_manifest.csv`, excluded invalid records, and `split_method.json` with input hashes and software versions. `source_row_id` is the zero-based row index in the source CSV, excluding its header. `type=SMILES` denotes the input representation, including peptides represented as SMILES; it is not a modality label.

These files were reconstructed from the archived development data and splitting code. The original checkpoint-associated graph cache was unavailable, so equivalence to historical checkpoint inputs is unverified.

The small-molecule input is `training_samples2.csv` (9,117 raw records; one empty structure excluded). The mixed input is `train_9.csv` (10,018 valid records). No additional sorting or deduplication is applied. The exporter checks the ordered structure-label hash before writing assignments.

```bash
python scripts/split_development_data.py --setting small --data /path/to/training_samples2.csv --output-dir new_splits/small_seed88
python scripts/split_development_data.py --setting mixed --data /path/to/train_9.csv --output-dir new_splits/mixed_seed42
```

Small-molecule seed88 is recorded in `scripts/train_plat.py` at revision `474d41d`; mixed seed42 is recorded in the archived [LiBP script](https://github.com/SHENXIN516/LiBP/blob/a7f76da7f273f9ae9a883c6a28ae3b5c80e60939/scripts/train_plat.py).

Figure 4's 200-record mixed holdouts and the 722/90/90 similarity-clustered peptide evaluation are separate designs and are not represented by these files. The Figure 4 holdout-generation script was not located in the available source archive.
