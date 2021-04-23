# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta
import argparse
from tqdm import tqdm

# Torch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.nn import functional as F

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

# Custom library
from utils import *
from features_type1 import generate_label, feature_engineering_type1
from features_type2 import generate_label, feature_engineering_type2
from features_NN import generate_label, feature_engineering_NN

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TOTAL_THRES = 300 # 구매액 임계값
SEED = 42 # 랜덤 시드
seed_everything(SEED) # 시드 고정


data_dir = '../input' # os.environ['SM_CHANNEL_TRAIN']
model_dir = '../model' # os.environ['SM_MODEL_DIR']
output_dir = '../output' # os.environ['SM_OUTPUT_DATA_DIR']

def make_lgb_oof_prediction(train, y, test, features, categorical_features='auto', model_params=None, folds=10):
    x_train = train[features]
    x_test = test[features]
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

        # LightGBM 데이터셋 선언
        dtrain = lgb.Dataset(x_tr, label=y_tr)
        dvalid = lgb.Dataset(x_val, label=y_val)
        
        # LightGBM 모델 훈련
        clf = lgb.train(
            model_params,
            dtrain,
            valid_sets=[dtrain, dvalid], # Validation 성능을 측정할 수 있도록 설정
            categorical_feature=categorical_features,
            verbose_eval=200
        )

        # Validation 데이터 예측
        val_preds = clf.predict(x_val)
        
        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 측정
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict(x_test) / folds
        
        # 폴드별 피처 중요도 저장
        fi[f'fold_{fold+1}'] = clf.feature_importance()

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력
        
    # 폴드별 피처 중요도 평균값 계산해서 저장 
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)

    
    return y_oof, test_preds, fi

def do_nn(train,test,y,features):
    # Dataset to Tensor
    train = train[features]
    test = test[features]

    train_t = torch.tensor(train.to_numpy(), dtype=torch.float32)
    y_t = torch.tensor(y.to_numpy(), dtype=torch.float32)
    test_t = torch.tensor(test.to_numpy(), dtype=torch.float32)

    train_len, test_len = len(train), len(test)

    # Setting
    N_EPOCH = 1000
    BATCH_SIZE = 128
    LOADER_PARAM = {
        'batch_size': BATCH_SIZE,
        'num_workers': 2,
    }
    # Seed Ensemble
    REPEATS=[42,777,111,31,17]
    for repeat in REPEATS:
        seed_everything(repeat)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=repeat)

        test_preds = np.zeros(test.shape[0])
        final_preds = np.zeros(test.shape[0])
        OOF_mean_score=0
        for fold, (tr_idx, val_idx) in enumerate(skf.split(train_t, y_t)):
            print(f'Fold : {fold}')
            print("-"*80)
            tr_idx, val_idx = list(tr_idx), list(val_idx)
            train_loader = DataLoader(TensorDataset(train_t[tr_idx, :], y_t[tr_idx]),
                                        shuffle=True, drop_last=False, **LOADER_PARAM)
            valid_loader = DataLoader(TensorDataset(train_t[val_idx, :], y_t[val_idx]),
                                    shuffle=False, drop_last=False, **LOADER_PARAM)
            test_loader = DataLoader(TensorDataset(test_t, torch.zeros((test_len,), dtype=torch.float32)),
                                    shuffle=False, drop_last=False, **LOADER_PARAM)

            model = nn.Sequential(
                    nn.Linear(252, 512, bias=False),
                    nn.LeakyReLU(0.05, inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256, bias=False),
                    nn.LeakyReLU(0.05, inplace=True),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128, bias=False),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64, bias=False),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.3),
                    nn.Linear(64, 1),
                    nn.Sigmoid()).to(device)

            criterion = torch.nn.BCEWithLogitsLoss().to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=4e-2)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=N_EPOCH // 4, eta_min=1.2e-5)
            early_stopping = EarlyStopping(patience = 50, verbose = False)
            fold_best_score = 0
            for epoch in range(N_EPOCH):
                model.train()
                train_acc, train_count = 0,0
                for idx,(train_x, train_y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    train_x, train_y = train_x.to(device), train_y.to(device)
                    pred = model(train_x).squeeze()

                    loss = criterion(pred, train_y)
                    loss.backward()
                    optimizer.step()
                    train_acc += ((pred > 0.5).float() == train_y).sum().item()
                    train_count += len(train_y)
                    scheduler.step(epoch + idx / len(train_loader))

                with torch.no_grad():
                    valid_true=[]
                    valid_pred=[]
                    model.eval()
                    running_acc, running_loss, running_count = 0, 0., 0
                    for valid_x, valid_y in valid_loader:
                        valid_x, valid_y = valid_x.to(device), valid_y.to(device)
                        pred = model(valid_x).squeeze()
                        loss = criterion(valid_y,pred)
                        running_loss += loss.item() * len(valid_y)
                        running_count += len(valid_y)
                        running_acc += ((pred > 0.5).float() == valid_y).sum().item()

                        valid_true.extend(valid_y.tolist())
                        valid_pred.extend(pred.tolist())

                valid_score = roc_auc_score(valid_true,valid_pred)
                early_stopping(valid_score, model)
                if fold_best_score<valid_score:
                    fold_best_score=valid_score
                if early_stopping.early_stop:
                    model.load_state_dict(torch.load('../checkpoint.pt'))
                    break
                if epoch%10==0:
                    print(f'[Epoch : {epoch}] Train Accuracy : {train_acc/train_count} Valid Accuracy : {running_acc/running_count} Valid AUC : {valid_score}')

            temp_pred=[]
            for test_x, _ in tqdm(test_loader):
                test_x=test_x.to(device)
                pred=model(test_x).squeeze()

                temp_pred.extend(pred.tolist())

            temp_pred=np.array(temp_pred)
            test_preds+=temp_pred/10.
            OOF_mean_score+=fold_best_score/10.
        final_preds+=test_preds/len(REPEATS)
        print("-"*80)
        print(f'Random Seed : {repeat} OOF_mean_AUC_score : {OOF_mean_score}')
        print("-"*80)
    
    # NN Submission
    sub = pd.read_csv('../input/sample_submission_type2.csv')
    final_sub = pd.read_csv('../input/sample_submission.csv')

    # 테스트 예측 결과 저장
    sub['probability'] = final_preds
    final_sub=final_sub.drop(['probability'],axis=1)
    final_sub = final_sub.merge(sub, on=['customer_id'],how='left')
    final_sub=final_sub.fillna(0.0)

    # 제출 파일 쓰기
    final_sub.to_csv('../output/output_NN_ensemble_type2.csv', index=False)

def do_lgbm(high_features,train,test,y,features,train_type='type1'):
    model_params = {
        'objective': 'binary', # 이진 분류
        'boosting_type': 'gbdt',
        'metric': 'auc', # 평가 지표 설정
        'feature_fraction': 0.8, # 피처 샘플링 비율
        'bagging_fraction': 0.8, # 데이터 샘플링 비율
        'bagging_freq': 1,
        'n_estimators': 10000, # 트리 개수
        'early_stopping_rounds': 100,
        'seed': SEED,
        'verbose': -1,
        'n_jobs': -1,
    }

    ensemble_preds = np.zeros(test.shape[0])
    repeat=float(len(high_features))
    for features in high_features:
        c_features=['customer_id','year_month','label']+features
        c_train=train[c_features]
        c_test=test[c_features]
        y_oof, test_preds, fi = make_lgb_oof_prediction(c_train, y, c_test, features, model_params=model_params)
        ensemble_preds+=test_preds/repeat

    # 2011-10월까지의 데이터로 2011-11월 예측 훈련 / 최종 2011-12월 예측 : train_type = type1
    if train_type=='type1':
        sub = pd.read_csv(data_dir + '/sample_submission.csv')
        sub['probability'] = ensemble_preds
        os.makedirs(output_dir, exist_ok=True)
        sub.to_csv(os.path.join(output_dir,'output_lgbm_ensemble_type1.csv'), index=False)
    # 2010-11월까지의 데이터로 2010-12월 예측 훈련 / 2010-12월부터 2011년-11월까지의 데이터로 최종 2011-12월 예측 : train_type = type2
    elif train_type=='type2':
        sub = pd.read_csv(data_dir + '/sample_submission_type2.csv')
        sub['probability'] = ensemble_preds
        final_sub = pd.read_csv(data_dir + '/sample_submission.csv')
        final_sub=final_sub.drop(['probability'],axis=1)
        final_sub = final_sub.merge(sub, on=['customer_id'],how='left')
        final_sub=final_sub.fillna(0.0)
        os.makedirs(output_dir, exist_ok=True)
        final_sub.to_csv(os.path.join(output_dir,'output_lgbm_ensemble_type2.csv'), index=False)

def get_submission():
    # 최종 제출 파일 생성
    preds1 = pd.read_csv('../output/output_lgbm_ensemble_type1.csv')['probability'] # type1 LGBM
    preds2 = pd.read_csv('../output/output_NN_ensemble_type2.csv')['probability'] # MLP
    preds3 = pd.read_csv('../output/output_lgbm_ensemble_type2.csv')['probability'] # type2 LGBM

    # 최종 Soft voting 앙상블 + Mixmatch
    preds = preds1**0.5 + (preds2**0.5+preds3**0.5)/2.

    sub = pd.read_csv('../input/sample_submission.csv')
    # 테스트 예측 결과 저장
    sub['probability'] = preds
    # 제출 파일 쓰기
    sub.to_csv('../output/output_final.csv', index=False)

if __name__ == '__main__':
    # 데이터 파일 읽기
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])
    # 예측할 연월 설정
    year_month = '2011-12'

    # 검증세트 1 : LGBM
    train, test, y, features = feature_engineering_type1(data, year_month)
    high_features=get_type1_features()
    do_lgbm(high_features,train,test,y,features,train_type='type1')

    # 검증세트 2 : LGBM
    train, test, y, features = feature_engineering_type2(data, year_month)
    high_features=get_type2_features()
    do_lgbm(high_features,train,test,y,features,train_type='type2')

    # 검증세트 2 : NN (MLP)
    train, test, y, features = feature_engineering_NN(data, year_month)
    do_nn(train,test,y,features)

    # 최종 제출 csv 파일
    get_submission()