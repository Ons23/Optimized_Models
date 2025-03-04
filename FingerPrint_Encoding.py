#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

class FingerprintGenerator:
    def __init__(self, model):
        self.model = model
        self.mols = None
        self.fp = None
    def set_molecules(self, data):
        self.mols = [Chem.MolFromSmiles(x) for x in data["SMILES"]]
    def generate_fingerprints(self):
        if self.model == "morg_fp":
            self.fp = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in self.mols]
        elif self.model == "morg_fp3":
            self.fp = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in self.mols]
        elif self.model == "rdk_fp":
            self.fp = [AllChem.RDKFingerprint(m) for m in self.mols]
        elif self.model == 'ap_fp':
            self.fp = [Chem.GetHashedAtomPairFingerprintAsBitVect(m) for m in self.mols]
        elif self.model == 'torsion_fp':
            self.fp = [Chem.GetHashedTopologicalTorsionFingerprintAsBitVect(m) for m in self.mols]

        fp_np = []
        for x_fp in self.fp:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(x_fp, arr)
            fp_np.append(arr)
        fp_np = np.array(fp_np).astype(np.int8)
        return (fp_np)


