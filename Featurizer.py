#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import deepchem as dc

class Featurizer:
    def __init__(self, model):
        self.model = model
        
    def featurize(self, data):
        if self.model in ["graphconv", "dag"]:
            featurizer = dc.feat.ConvMolFeaturizer() 
        elif self.model in ["gat", "gcn"]:
            featurizer = dc.feat.MolGraphConvFeaturizer()
        elif self.model in ["afp", "mpnn"]:
            featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
            
        tasks = ['label']
        loader = dc.data.CSVLoader(tasks=tasks, feature_field="SMILES", featurizer=featurizer)
        dataset = loader.create_dataset(data, data_dir= "{}_dataset".format(self.model))
        transformer = None
        
        return dataset, [transformer]

