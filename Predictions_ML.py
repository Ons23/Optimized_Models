import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, f1_score

class Predict :
    def __init__ (self, model_name) :
        self.model_name= model_name
    def predictions (self, model,data,x_test, y_test):
        test_scores =[]
        pred_test = model.predict(x_test)
        test_scores.append(accuracy_score(y_test, pred_test))
        test_scores.append(precision_score(y_test, pred_test))
        test_scores.append(recall_score (y_test, pred_test))
        test_scores.append(f1_score(y_test, pred_test))
        test_scores.append (matthews_corrcoef (y_test, pred_test))
        test_scores.append(balanced_accuracy_score (y_test, pred_test))
        test_scores.append(roc_auc_score(y_test,model.predict_proba(x_test)[:, 1]))
        cm = confusion_matrix(y_test, pred_test)
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn)  
        fpr = fp / (fp + tn) 

        test_scores.append(tpr)
        test_scores.append(fpr)

        ConfusionMatrixDisplay.from_predictions(y_test, pred_test, cmap="Blues")
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.title('External dataset', size=15)
        filename = f"{self.model_name}_confusion_matrix_test.png"
        plt.savefig(filename, dpi=300)
        
        metric_names = ["Accuracy", "Precision", "Recall", "F1_score", "MCC", "Balanced_Accuracy", "ROC-AUC", "tpr", "fpr"]        
        test_df = pd.DataFrame({"values": test_scores, "Metrics": metric_names})
        test_df["dataset"] = "test_scores"
        test_df = test_df.pivot(index="Metrics", columns="dataset", values="values")
        test_df = test_df.reset_index()
        test_df.insert(0, "Model Name", self.model_name)
        test_df.to_csv('{}_Perfromance_metrics.csv'.format(self.model_name), index=False)
        
        predicted_probabilities = model.predict_proba(x_test)
        predictions = pred_test.tolist()
        my_pred_prob = predicted_probabilities.tolist()
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
        results["label"] = data ['label']
        results["Activity"] = data["Activity"]
        results["Prediction"] = predictions
        results["Activity_prediction"] = activity
        results["Probability"] = probability
        results
        results.to_csv('{}_Predictions_Results.csv'.format(self.model_name), index=False)       
        return (results)