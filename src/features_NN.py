import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Custom library
from utils import seed_everything, print_score


TOTAL_THRES = 300 # 구매액 임계값
SEED = 42 # 랜덤 시드
seed_everything(SEED) # 시드 고정

data_dir = '../input/train.csv' # os.environ['SM_CHANNEL_TRAIN']
model_dir = '../model' # os.environ['SM_MODEL_DIR']

def get_prev_ym(year_month):
    # year_month 이전 년 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(years=1)
    prev_ym = prev_ym.strftime('%Y-%m')
    return prev_ym
'''
    입력인자로 받는 year_month에 대해 고객 ID별로 총 구매액이
    구매액 임계값을 넘는지 여부의 binary label을 생성하는 함수
'''
def generate_label(df, year_month, total_thres=TOTAL_THRES, print_log=True):
    prev_ym = get_prev_ym(year_month)

    df = df.copy()
    # year_month에 해당하는 label 데이터 생성
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
    df.reset_index(drop=True, inplace=True)

    # year_month 이전 월의 고객 ID 추출
    # modify
    cust = df[(df['year_month']>=prev_ym) & (df['year_month']<year_month)]['customer_id'].unique()
    # year_month에 해당하는 데이터 선택
    df = df[df['year_month']==year_month]
    
    # label 데이터프레임 생성
    label = pd.DataFrame({'customer_id':cust})
    label['year_month'] = year_month
    
    # year_month에 해당하는 고객 ID의 구매액의 합 계산
    grped = df.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    
    # label 데이터프레임과 merge하고 구매액 임계값을 넘었는지 여부로 label 생성
    label = label.merge(grped, on=['customer_id','year_month'], how='left')
    label['total'].fillna(0.0, inplace=True)
    label['label'] = (label['total'] > total_thres).astype(int)

    # 고객 ID로 정렬
    label = label.sort_values('customer_id').reset_index(drop=True)
    if print_log: print(f'{year_month} - final label shape: {label.shape}')
    return label


def feature_preprocessing(train, test, features, do_imputing=True):
    x_tr = train.copy()
    x_te = test.copy()
    
    # 범주형 피처 이름을 저장할 변수
    cate_cols = []

    # 레이블 인코딩
    for f in features:
        if x_tr[f].dtype.name == 'object': # 데이터 타입이 object(str)이면 레이블 인코딩
            cate_cols.append(f)
            le = LabelEncoder()
            # train + test 데이터를 합쳐서 레이블 인코딩 함수에 fit
            le.fit(list(x_tr[f].values) + list(x_te[f].values))
            
            # train 데이터 레이블 인코딩 변환 수행
            x_tr[f] = le.transform(list(x_tr[f].values))
            
            # test 데이터 레이블 인코딩 변환 수행
            x_te[f] = le.transform(list(x_te[f].values))

    print('categorical feature:', cate_cols)

    if do_imputing:
        # 중위값으로 결측치 채우기
        imputer = SimpleImputer(strategy='median')

        x_tr[features] = imputer.fit_transform(x_tr[features])
        x_te[features] = imputer.transform(x_te[features])
    
    return x_tr, x_te

def make_feature(df,year_month):
    # 고객 별 모든 월 별 total, price, quantity에 대해서 agg_func 적용
    prev_ym=get_prev_ym(year_month)
    df = df.copy()
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')

    cust = df[(df['year_month']>=prev_ym) & (df['year_month']<year_month)]['customer_id'].unique()
    month_df = pd.DataFrame({'customer_id':cust})

    agg_func = ['min','max','sum','count','std','skew','mean']
    train_year_month=['2009-12','2010-01','2010-02','2010-03','2010-04','2010-05','2010-06','2010-07','2010-08','2010-09','2010-10','2010-11','2010-12',
                    '2011-01','2011-02','2011-03','2011-04','2011-05','2011-06','2011-07','2011-08','2011-09','2011-10','2011-11']
    
    count=0
    for tr_ym in train_year_month:
        if (tr_ym>=prev_ym) & (tr_ym<year_month):
            count+=1  
            train_agg = df.loc[df['year_month'] == tr_ym].groupby(['customer_id'])['total','price','quantity'].agg(agg_func)
            new_cols = []
            for col in train_agg.columns.levels[0]:
                for stat in train_agg.columns.levels[1]:
                    new_cols.append(f'{count}-{col}-{stat}')
            train_agg.columns = new_cols
            train_agg.reset_index(inplace = True)

            month_df=month_df.merge(train_agg, on=['customer_id'], how='left')
    month_df.fillna(0.0, inplace=True)

    return month_df

def feature_engineering_NN(df, year_month):
    df = df.copy()
    
    # year_month 이전 년도
    prev_ym=get_prev_ym(year_month)
    
    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id','year_month','label']]
    test_label = generate_label(df, year_month)[['customer_id','year_month','label']]
    
    train_feature = make_feature(df,prev_ym)
    test_feature = make_feature(df,year_month)

    all_train_data = train_label.merge(train_feature,on=['customer_id'], how='left')
    test_data = test_label.merge(test_feature,on=['customer_id'], how='left')

    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, all_train_data['label'], features


if __name__ == '__main__':
    
    print('data_dir', data_dir)
