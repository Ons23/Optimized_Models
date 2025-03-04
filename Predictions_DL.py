#!/usr/bin/env python
# coding: utf-8

# In[2]:
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import dgllife
import torch.nn
import deepchem as dc
from deepchem.metrics.metric import Metric
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

class Predict :
    def __init__ (self, model_name) :
        self.model_name= model_name
    def predictions (self, model,data ,test_dataset,test_dataset_y,test_scores):
        y_test_pred = model.predict(test_dataset)
        y_tst_pred = np.argmax(y_test_pred, axis=1) 
        label = np.expand_dims(y_tst_pred, -1)


        precision_scores = [self.model_name,"Precision"]
        mcc_scores = [self.model_name,"MCC"]
        balanced_acc_scores = [self.model_name,"Balanced Accuracy"]
        
        precision_val = precision_score(test_dataset_y, label, average='binary')
        precision_scores.append(precision_val)
        
        mcc_val = matthews_corrcoef(test_dataset_y, label)
        mcc_scores.append(mcc_val)
        
        balanced_acc_val = balanced_accuracy_score(test_dataset_y, label)
        balanced_acc_scores.append(balanced_acc_val)

        # Compute TPR and FPR
        cm = confusion_matrix(test_dataset_y, label)
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn)  # True Positive Rate
        fpr = fp / (fp + tn)  # False Positive Rate
        
        tpr_scores = [self.model_name, "tpr", tpr]
        fpr_scores = [self.model_name, "fpr", fpr]

        cr= classification_report(test_dataset_y, label)
        ConfusionMatrixDisplay.from_predictions(test_dataset_y, label, cmap = "Blues")
        all_sample_title = "External Dataset"
        plt.title(all_sample_title, size = 15)
        filename = "{}_confusion_matrix_external.png".format(self.model_name)
        plt.savefig(filename, dpi=300)

        model_names = []
        metric_names = []
        test_scores_list = []
        for model_name, metrics in test_scores.items():
            for metric_name, metric_value in metrics.items():
                model_names.append(model_name)
                metric_names.append(metric_name.replace("metric-1", "ROC-AUC").replace("metric-2", "Accuracy").replace("metric-3", "F1-Score").replace("metric-4", "Recall"))
                test_scores_list.append(metric_value)
        
        metric_results = pd.DataFrame({'Model': model_names, 'Metrics': metric_names, 'test_score': test_scores_list})
        metric_results.loc[len(metric_results)] = precision_scores
        metric_results.loc[len(metric_results)] = mcc_scores
        metric_results.loc[len(metric_results)] = balanced_acc_scores
        metric_results.loc[len(metric_results)] = tpr_scores
        metric_results.loc[len(metric_results)] = fpr_scores
        metric_results.to_csv('{}_Perfromance_metrics.csv'.format(self.model_name), index=False)

        predictions = y_tst_pred.tolist()
        my_pred_prob = y_test_pred.tolist()
        probability = []
        activity = []

        for index, prob in zip(predictions, my_pred_prob):
            probability.append(round(prob[index], 3))
            if index==0:
                activity.append("inactive")
            else:
                activity.append("active")
        
        results = pd.DataFrame()
        results["SMILES"] = data["SMILES"]
        results["Label"] = data ['label']
        results["Activity"] = data["Activity"]
        results["Prediction"] = predictions
        results["Activity_prediction"] = activity
        results["Probability"] = probability
        
        results.to_csv('{}_Predictions_Results.csv'.format(self.model_name), index=False)       
        return results
