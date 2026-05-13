import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch_geometric.data import Dataset, Data, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split  

from rdkit import Chem
from rdkit.Chem import AllChem
sys.path.append('...')
from plat_model.model import SubGT, GraphTransformer 


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"Input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def calc_atom_features(atom, explicit_H=False):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']
    ) + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
           [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
           one_of_k_encoding_unk(atom.GetHybridization(), [
               Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
               Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
               Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    return np.array(results)


def calc_bond_features(bond, use_chirality=False):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(), bond.IsInRing()
    ]
    if use_chirality:
        bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = np.array([calc_atom_features(a) for a in mol.GetAtoms()])

    row, col, edge_attr = [], [], []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        bond_feats = calc_bond_features(bond)

        row += [a, b]
        col += [b, a]

        edge_attr.append(bond_feats)
        edge_attr.append(bond_feats)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)


class BBBP_Dataset(Dataset):
    def __init__(self, csv_path, cache_dir="..."):
        super().__init__()
        self.cache_dir = cache_dir
        self.csv_path = csv_path

        cache_file = os.path.join(self.cache_dir, "bbbp_graphs.pt")
        if os.path.exists(cache_file):
            print(f"Loading preprocessed data from {cache_file}")
            data = torch.load(cache_file)
            self.smiles = data["smiles"]
            self.labels = data["labels"]
            self.graphs = data["graphs"]
        else:
            df = pd.read_csv(self.csv_path)
            df = df[df["type"] == "SMILES"]

            smiles = df["sequence"].astype(str).tolist()
            labels = df["label"].astype(int).tolist()

            self.graphs = []
            self.labels = []
            self.smiles = []
            for smi, label in zip(smiles, labels):
                g = mol_to_graph(smi)
                if g is not None:
                    self.graphs.append(g)
                    self.labels.append(label)
                    self.smiles.append(smi)

            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save({"smiles": self.smiles, "labels": self.labels, "graphs": self.graphs}, cache_file)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        graph = self.graphs[idx]
        graph.y = torch.tensor([self.labels[idx]], dtype=torch.float)
        return graph


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)  

        loss = criterion(out, batch.y.long())  
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, trues, probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)

            out = torch.softmax(out, dim=-1)

            pred = torch.argmax(out, dim=-1).cpu().numpy()  
            probs_batch = out[:, 1].cpu().numpy()  

            preds.extend(pred)
            trues.extend(batch.y.cpu().numpy())
            probs.extend(probs_batch)  

    preds = torch.tensor(preds)
    trues = torch.tensor(trues)

    auc = roc_auc_score(trues.numpy(), probs)

    f1 = f1_score(trues.numpy(), preds.numpy())

    mcc = matthews_corrcoef(trues.numpy(), preds.numpy())

    acc = (preds == trues).float().mean().item()

    return auc, f1, mcc, acc


def main():
    csv_path = ".../dataset/SMILES.csv"  
    dataset = BBBP_Dataset(csv_path)

    train_smiles, temp_smiles, train_labels, temp_labels = train_test_split(
        dataset.smiles, dataset.labels, test_size=0.2, random_state=88)

    val_smiles, test_smiles, val_labels, test_labels = train_test_split(
        temp_smiles, temp_labels, test_size=0.5, random_state=88)

    train_dataset = [dataset.get(i) for i in range(len(dataset)) if dataset.smiles[i] in train_smiles]
    val_dataset = [dataset.get(i) for i in range(len(dataset)) if dataset.smiles[i] in val_smiles]
    test_dataset = [dataset.get(i) for i in range(len(dataset)) if dataset.smiles[i] in test_smiles]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphTransformer(
        in_channels=38,
        edge_features=6,
        num_hidden_channels=256,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.9, verbose=True)

    epochs = 500  
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)

        auc_test, f1_test, mcc_test, acc_test = evaluate(model, test_loader, device)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {loss}, Test AUC: {auc_test:.4f}, Test F1-Score: {f1_test:.4f}, Test MCC: {mcc_test:.4f}, "
              f"Test Accuracy: {acc_test:.4f}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        print("-" * 80)

        scheduler.step(acc_test)


if __name__ == "__main__":
    main()
