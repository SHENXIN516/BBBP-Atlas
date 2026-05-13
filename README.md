# BBBP-Atlas

BBBP-Atlas is a curated blood–brain barrier permeability (BBBP) resource and graph-based molecular learning framework for BBB permeability analysis, descriptor validation, and interpretable cheminformatics research.

The repository integrates standardized BBB-related datasets, RDKit-based molecular preprocessing, graph neural network utilities, and reproducible benchmark pipelines for BBB permeability modeling. It is designed to support both dataset-centric exploration and graph-based machine learning experiments in CNS drug discovery.

An interactive web platform for molecular browsing and dataset exploration is publicly available at:

https://cadd.drugflow.com/bbbp/

---

## Highlights

- Curated BBB permeability datasets in JSON and CSV formats
- Standardized RDKit preprocessing and descriptor generation
- Graph-based molecular representation learning with PyTorch Geometric
- Reproducible training and evaluation pipeline
- Support for both small molecules and peptide-related BBB datasets
- Interactive web platform for visualization and molecular exploration

---

## Repository Structure

```text
BBBP-Atlas/
├── dataset/              # BBBP datasets
├── plat_model/           # Graph neural network modules
├── scripts/              # Training and preprocessing scripts
├── ckpt/                 # Model checkpoints and outputs
├── environment.yml       # Conda environment definition
└── README.md
```

---

## Included Datasets

All datasets are stored under `dataset/`.

Current dataset variants include:

- `bbbp_benchmark.json`
- `bbbp_benchmark.csv`
- `bbbp_quantitative.json`
- `bbbp_small_molecule.json`
- `bbbp_small_molecule.csv`
- `bbbp_peptide.json`
- `bbbp_peptides.csv`

The datasets are organized for downstream molecular property prediction, graph construction, and benchmark evaluation.

---

## Molecular Preprocessing

Molecular graphs are constructed using RDKit and converted into PyTorch Geometric graph objects.

Atom-level features currently include atom type, degree, formal charge, radical electrons, hybridization state, aromaticity, and hydrogen count. Bond-level features include bond type, conjugation, ring membership, and optional stereochemical information.

---

## Model Pipeline

The main training entry point is:

```bash
python scripts/train_plat.py
```

The pipeline performs molecular parsing from SMILES strings, graph construction with RDKit, dataset splitting, graph caching, graph-based model training, and evaluation using multiple classification metrics.

Reported metrics include:

- ROC-AUC
- Accuracy
- F1-score
- MCC
- BA
- SE
- SP

The current implementation is intended as a reproducible graph-based BBBP benchmark pipeline and lightweight research framework for molecular learning experiments.

---

## Installation

We recommend using Conda for environment management.

```bash
conda env create -f environment.yml
conda activate bbbp-atlas
```

If dependency resolution is slow, `mamba` is recommended:

```bash
mamba env create -f environment.yml
```

---

## Quick Start

After activating the environment:

```bash
python scripts/train_plat.py
```

Before training, update the dataset and cache paths in the configuration section of the script if necessary.

The pipeline will automatically preprocess molecules, cache graph objects, split the dataset, train the model, and report evaluation metrics.

---

## Data Format

The CSV-based training pipeline expects at least the following columns:

- `type`
- `sequence`
- `label`

Rows with `type == "SMILES"` are interpreted as molecular entries.

JSON datasets follow a unified molecular record structure containing SMILES strings, molecular descriptors, task annotations, and associated metadata.

---

## Reproducibility Notes

Descriptor values may vary slightly depending on the RDKit version, descriptor implementation, and molecular standardization strategy. For strict reproducibility, we recommend using the provided Conda environment and preserving the original preprocessing workflow.

---

## Web Platform

BBBP-Atlas also provides an interactive web interface for molecular browsing and dataset exploration:

https://cadd.drugflow.com/bbbp/

The platform is intended to support rapid inspection of BBB-related molecular records and facilitate lightweight interactive analysis.

---

## Citation

If you use BBBP-Atlas in your research, please cite the corresponding publication and repository.

```bibtex
@article{BBBPAtlas2026,
  title={BBBP-Atlas: Unified Interpretable Modeling of BBB Permeability across Small Molecules and Peptides},
  author={Xin Shen, Qun Su, Hao Luo, Qiaolin Gou, Jingxuan Ge, Jike Wang, Yu Kang, Tingjun Hou},
  journal={...},
  year={2026}
}
```

---

## License

This project is released under the MIT License.

Third-party datasets and dependencies retain their original licenses.
