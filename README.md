# BBBP-Atlas

BBBP-Atlas is a curated blood–brain barrier permeability (BBBP) resource and cheminformatics analysis framework designed for molecular property exploration, descriptor validation, and interpretable BBB permeability modeling.

The repository integrates standardized BBB-related datasets, RDKit-based descriptor computation pipelines, validation utilities, and multidimensional visualization modules for molecular representation analysis and benchmark exploration.

---

## Features

### Curated BBBP Datasets
- Standardized BBB permeability datasets in JSON and CSV formats
- Multiple dataset variants, including:
  - benchmark datasets
  - qualitative BBB annotations
  - peptide-related subsets
- Unified molecular metadata and descriptor annotations

### RDKit-Based Descriptor Computation
Computed molecular descriptors include:

- Canonical SMILES
- Molecular weight (`MolWt` / `ExactMolWt`)
- LogP (`Crippen.MolLogP`)
- Topological polar surface area (TPSA)
- Hydrogen bond donors/acceptors (HBD/HBA)
- Rotatable bonds
- Aromatic ring counts

### Descriptor Validation Utilities
The repository provides automated validation scripts for consistency checking between stored descriptors and RDKit recomputation.
