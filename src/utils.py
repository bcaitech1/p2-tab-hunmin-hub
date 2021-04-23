import torch
import os
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# 시드 고정 함수
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# 평가 지표 출력 함수
def print_score(label, pred, prob_thres=0.5):
    print('Precision: {:.5f}'.format(precision_score(label, pred>prob_thres)))
    print('Recall: {:.5f}'.format(recall_score(label, pred>prob_thres)))
    print('F1 Score: {:.5f}'.format(f1_score(label, pred>prob_thres)))
    print('ROC AUC Score: {:.5f}'.format(roc_auc_score(label, pred)))

# EarlyStopping
class EarlyStopping:
    def __init__(self, patience=40, verbose=False, delta=0, path='../checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_auc, model):

        score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print(f'Best Valid AUC Score : {self.val_auc_max}')
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_auc_max = val_auc

# data_type1 : lgbm / Best Features
def get_type1_features():
    high_features=[]
    high_features.append(['total-max', 'total-sum', 'price-max', 'price-min', 
                'Recency', 'Frequency', 'RecencyScore', 'FrequencyScore', 'RFM_SCORE', 'avg_order_value', 'purchase_frequency', 'profit_margin', 'cv', 'cltv', 
                'december_total_09', 'december_total_10', 
                'all_mean', 'live_mean', 
                'cohort','gap_month'])
    high_features.append(['total-max', 'total-min', 'total-sum', 'price-max', 'price-min', 'price-sum', 
                'Recency', 'Frequency', 
                'december_total_09', 'december_total_10', 
                'all_mean', 'cohort'])
    high_features.append(['total-max', 'total-min', 'total-sum', 'price-max', 'price-min', 'price-sum', 
                'Recency', 'Frequency', 'Monetary', 'RecencyScore', 'FrequencyScore', 'MonetaryScore', 
                'avg_order_value', 'profit_margin', 'cv', 'cltv', 
                'december_total_10', 
                'all_mean', 'live_mean', 
                'cohort', 'gap_month'])
    high_features.append(['total-max', 'total-min', 'price-max', 'price-min', 'price-sum', 
                'Recency', 'Frequency', 'Monetary', 'RecencyScore', 'FrequencyScore', 'MonetaryScore', 
                'purchase_frequency', 'profit_margin', 'cv', 'cltv', 
                'december_total_09', 'december_total_10', 
                'all_mean', 'live_mean', 
                'cohort', 'gap_month'])
    high_features.append(['total-max', 'total-sum', 'price-max', 'price-min', 
                'Recency', 'Frequency', 'RecencyScore', 'FrequencyScore', 'RFM_SCORE', 'avg_order_value', 'purchase_frequency', 'profit_margin', 'cv', 'cltv', 
                'december_total_09', 'december_total_10', 
                'all_mean', 'live_mean', 
                'cohort', 'gap_month'])
    return high_features

# data_type2 : lgbm / Best Features
def get_type2_features():
    high_features=[]
    high_features.append(['total-max', 'total-min', 'total-sum', 'price-max', 'price-min', 'price-sum', 
                    'Recency', 'Frequency', 'Monetary', 'RecencyScore', 'FrequencyScore', 'MonetaryScore', 'RFM_SCORE', 
                    'purchase_frequency', 'cltv', 
                    'all_mean', 'live_mean', 
                    'prev_december', 'over_rate', 'gap_month'])
    high_features.append(['total-max', 'total-min', 'total-sum', 'price-max', 'price-min', 'price-sum', 
                    'Recency', 'Frequency', 'Monetary', 'FrequencyScore', 'MonetaryScore', 
                    'purchase_frequency', 'profit_margin', 'cv', 'cltv', 
                    'all_mean', 'prev_december', 'over_rate', 'gap_month']) 
    return high_features
