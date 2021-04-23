import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from operator import attrgetter
from collections import defaultdict

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

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
        if x_tr[f].dtype.name == 'object' or x_tr[f].dtype.name == 'period[M]': # 데이터 타입이 object(str)이면 레이블 인코딩
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

def get_rfm(df,year_month):
    prev_ym = get_prev_ym(year_month)
    df=df.copy()

    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')

    df=df[(df['order_date']>=prev_ym) & (df['order_date'] < year_month)]

    if year_month=='2010-12':
        today_date = datetime.datetime(2010, 12,1)
    else:
        today_date = datetime.datetime(2011, 12,1)
    
    rfm = df.groupby('customer_id').agg({'order_date': lambda date: (today_date - date.max()).days,
                                     'order_id': lambda orders: orders.nunique(),
                                     'total': lambda total: total.sum()})
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    # 고객들을 20% 씩 그룹화
    rfm["RecencyScore"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    # 각 Category들을 합쳐 새로운 Category 생성
    rfm["RFM_SCORE"] = (rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str) + rfm['MonetaryScore'].astype(str))

    rfm = rfm.sort_values('customer_id')
    # 평균 구매 가치
    rfm['avg_order_value'] = rfm['Monetary'] / rfm['Frequency']
    # 평균 구매 빈도율
    rfm["purchase_frequency"] = rfm['Frequency'] / rfm.shape[0]
    # 이익율 (5% 가정)
    rfm['profit_margin'] = rfm['Monetary'] * 0.05
    # 재구매율과 이탈율
    repeat_rate = rfm[rfm.Frequency > 1].shape[0] / rfm.shape[0]
    churn_rate = 1 - repeat_rate
    # 고객 가치
    rfm['cv'] = (rfm['avg_order_value'] * rfm["purchase_frequency"])
    # 고객 평생 가치
    rfm['cltv'] = (rfm['cv'] / churn_rate) * rfm['profit_margin']
    rfm=rfm.reset_index()
    return rfm

def get_december_info(data,year_month):
    # 전 년도 12월의 고객 별 구매 액수
    prev_ym = get_prev_ym(year_month)

    df=data.copy()
    df=df[(df['order_date']>=prev_ym) & (df['order_date'] < year_month)]
    cust = df['customer_id'].unique()

    if year_month=='2010-12':
        df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
        df=df[df['year_month']=='2009-12']
        december = df.groupby('customer_id').agg({'total': lambda total: total.sum()})
        december.columns=['prev_december']
        december = december.sort_values('customer_id')
        december = december.reset_index()
    elif year_month=='2011-12':
        df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
        df=df[df['year_month']=='2010-12']
        december = df.groupby('customer_id').agg({'total': lambda total: total.sum()})
        december.columns=['prev_december']
        december = december.sort_values('customer_id')
        december = december.reset_index()

    customer = pd.DataFrame({'customer_id':cust})
    customer = customer.merge(december,on=['customer_id'],how='left')
    customer['prev_december'].fillna(0.0, inplace=True)
    customer = customer.sort_values('customer_id').reset_index(drop=True)
    return customer

def buy_month_mean(df,year_month):
    # all mean : 고객이 유입된 시점부터 학습 기준 일까지의 고객 별 평균 구매액
    # live mean : 고객이 유입된 시점부터 가장 최근까지 구매가 일어났던 날까지의 고객 별 평균 구매액
    prev_ym = get_prev_ym(year_month)

    def month_diff(a, b):
        if type(a) is not str:
            a=pd.DatetimeIndex(a)
        else:
            a=datetime.datetime.strptime(year_month, "%Y-%m")
        b=pd.DatetimeIndex(b)
        return 12 * (a.year - b.year) + (a.month - b.month)
    
    df=df.copy()
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
    cust = df[(df['year_month']>=prev_ym) & (df['year_month']<year_month)]['customer_id'].unique()
    output = pd.DataFrame({'customer_id':cust})

    df=df[(df['order_date']>=prev_ym) & (df['order_date'] < year_month)]

    live_buy = df.drop_duplicates('customer_id', keep='first')[['order_date','customer_id']].sort_values('customer_id').reset_index(drop=True)
    last_buy = df.drop_duplicates('customer_id', keep='last')[['order_date','customer_id']].sort_values('customer_id').reset_index(drop=True)

    live_buy.columns=['first_year_month','customer_id']
    last_buy.columns=['last_year_month','customer_id']

    live_buy=live_buy.merge(last_buy, on=['customer_id'],how='left')
    live_buy=live_buy[['customer_id','first_year_month','last_year_month']]
    live_buy['first_year_month']=live_buy['first_year_month'].dt.strftime('%Y-%m') #첫구매
    live_buy['last_year_month']=live_buy['last_year_month'].dt.strftime('%Y-%m') #마지막구매

    live_buy['live_month']=month_diff(live_buy['last_year_month'],live_buy['first_year_month'])
    live_buy['all_month']=month_diff(year_month,live_buy['first_year_month'])
    data_agg_sum = df.groupby('customer_id').sum().reset_index().sort_values('customer_id')
    # 고객 별 모든 거래량을 첫 구매로 부터 기준일까지로 나눔
    live_buy['all_mean']=data_agg_sum['total']/(live_buy['all_month']+1)
    # 고객 별 모든 거래량을 첫 구매로 부터 마지막 구매일까지로 나눔
    live_buy['live_mean']=data_agg_sum['total']/(live_buy['live_month']+1)

    output=output.merge(live_buy, on=['customer_id'], how='left')
    output=output.fillna(0)
    return output[['customer_id','all_mean','live_mean','live_month']]

def over_counting(df,year_month):
    # over_rate : 고객이 월 별 총 구매액이 300이 넘는 달 수 / 고객이 활동한 달 수
    prev_ym = get_prev_ym(year_month)
    df = df.copy()
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')

    cust = df[(df['year_month']>=prev_ym) & (df['year_month']<year_month)]['customer_id'].unique()
    month_df = pd.DataFrame({'customer_id':cust})

    agg_func = ['sum']
    train_year_month=['2009-12','2010-01','2010-02','2010-03','2010-04','2010-05','2010-06','2010-07','2010-08','2010-09','2010-10','2010-11','2010-12',
                    '2011-01','2011-02','2011-03','2011-04','2011-05','2011-06','2011-07','2011-08','2011-09','2011-10','2011-11']
    for i, tr_ym in enumerate(train_year_month):
        if (tr_ym>=prev_ym) & (tr_ym<year_month):
            # group by aggretation 함수로 train 데이터 피처 생성
            train_agg = df.loc[df['year_month'] == tr_ym].groupby(['customer_id'])['total'].agg(agg_func)
            new_cols = []
            for col in train_agg.columns:
                new_cols.append(f'{tr_ym}-{col}')
            train_agg.columns = new_cols
            train_agg[new_cols]=train_agg[new_cols]>300
            train_agg.reset_index(inplace = True)
            
            month_df=month_df.merge(train_agg, on=['customer_id'], how='left')
    
    month_df.fillna(0.0, inplace=True)
    month_df['over_count']=0
    for year in train_year_month:
        if (year>=prev_ym) & (year<year_month):
            current_col=year+'-sum'
            month_df['over_count']+=month_df[current_col]
    month_df = month_df.sort_values('customer_id').reset_index(drop=True)
    return month_df[['customer_id','over_count']]

def make_cohort(data,year_month,value_column='customer_id',agg_func='nunique',want_heatmap=True):
    prev_ym = get_prev_ym(year_month)

    # 기준 년-월(year_month) 이전의 고객 정보들을 모두 저장
    df=data[(data['order_date']>=prev_ym) & (data['order_date']<year_month)]
    # 사용할 columns : 고객id, 고객별 구매 일자, 건당 구매 액수
    df = df[['customer_id', 'order_date', 'total']]
    # 고객 별 구매 일자를 구매 년-월로 변환
    df['buy_month'] = df['order_date'].dt.to_period('M')
    # 고객 별 첫 구매한 년-월 획득
    df['cohort'] = df.groupby('customer_id')['order_date'].transform('min').dt.to_period('M')
    # cohort table 생성 : 고객id, 고객별 첫 구매 년-월, 고객별 구매 년-월
    df_cohort = df.groupby(['customer_id','cohort','buy_month']).agg(aggregate_value=('total', 'sum')).sort_values('customer_id').reset_index(drop=False)
    # comlumns name 재 설정
    df_cohort.columns=['customer_id','cohort','buy_month','total_month']
    # 고객 별 구매 년-월과 고객 별 첫 구매 년-월의 차이 계산 : buy_month - cohort
    df_cohort['gap_month'] = (df_cohort['buy_month'] - df_cohort['cohort']).apply(attrgetter('n'))
    
    # cohort pivot table 생성 (value_column= default: 'customer_id') , (agg_func= default: 'nunique')
    cohort_table=df_cohort.pivot_table(index = 'cohort',columns = 'gap_month',values = value_column,aggfunc = agg_func)
    
    # cohort 계산의 기준이 될 첫 구매 년-월의 정보
    cohort_size = cohort_table.iloc[:,0]
    cohort_matrix = cohort_table.divide(cohort_size, axis=0) # 기준(첫 구매 년-월)의 수치로 나누기
    # 시각화 (heat_map)
    if want_heatmap:
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 2, figsize=(16, 12), sharey=True, gridspec_kw={'width_ratios': [1, 11]})

            # cohort_matrix
            sns.heatmap(cohort_matrix,annot=True,fmt='.0%',cmap='RdPu',ax=ax[1])

            table_title=f'{value_column} : Cohort Analysis' # 제목 설정
            ax[1].set_title(table_title, fontsize=16)
            ax[1].set(xlabel='Periods',ylabel='')

            # cohort size
            cohort_size_df = pd.DataFrame(cohort_size).rename(columns={0: 'cohort_size'})
            sns.heatmap(cohort_size_df,annot=True,cbar=False,fmt='g',cmap='RdPu',ax=ax[0])
            fig.tight_layout()
    
    cohort_table=cohort_table.fillna(0.0)

    # get features
    feature_df = df_cohort.groupby(['customer_id','cohort']).mean().sort_values('customer_id').reset_index()
    return feature_df[['customer_id','cohort','gap_month']]

def feature_engineering_type2(df, year_month):
    df = df.copy()
    # 작년
    prev_ym = get_prev_ym(year_month)
    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[(df['order_date']>=prev_ym) & (df['order_date'] < year_month)]
    
    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id','year_month','label']]
    test_label = generate_label(df, year_month)[['customer_id','year_month','label']]

    # RFM table 생성
    train_rfm = get_rfm(df,prev_ym)
    test_rfm = get_rfm(df,year_month)
    print('train_rfm.shape', train_rfm.shape, ', test_rfm.shape', test_rfm.shape)

    # Buy month Mean 생성
    train_buy_mean=buy_month_mean(df,prev_ym)
    test_buy_mean=buy_month_mean(df,year_month)
    print('train_buy_mean.shape', train_buy_mean.shape, ', test_buy_mean.shape', test_buy_mean.shape)

    # December Info Table 생성
    train_december=get_december_info(df,prev_ym)
    test_december=get_december_info(df,year_month)
    print('train_december.shape', train_december.shape, ', test_december', test_december.shape)

    # Over count feature
    train_over=over_counting(df,prev_ym)
    test_over=over_counting(df,year_month)
    print('train_over.shape', train_over.shape, ', test_over.shape', test_over.shape)

    # Cohort feature 생성
    train_cohort=make_cohort(df,prev_ym,value_column='customer_id',agg_func='nunique',want_heatmap=False)
    test_cohort=make_cohort(df,year_month,value_column='customer_id',agg_func='nunique',want_heatmap=False)
    print('train_cohort.shape', train_cohort.shape, ', test_cohort.shape', test_cohort.shape)

    # group by aggregation 함수 선언
    agg_func = ['max','min','sum']
    all_train_data = pd.DataFrame()
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        # modify
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id'])['total','price'].agg(agg_func)

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for col in train_agg.columns.levels[0]:
            for stat in train_agg.columns.levels[1]:
                new_cols.append(f'{col}-{stat}')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        all_train_data = all_train_data.append(train_agg)
    
    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    # group by aggretation 함수로 test 데이터 피처 생성
    test_agg = test.groupby(['customer_id'])['total','price'].agg(agg_func)
    test_agg.columns = new_cols
    
    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')

    # RFM Feature 추가
    all_train_data = all_train_data.merge(train_rfm, on=['customer_id'], how='left')
    test_data = test_data.merge(test_rfm, on=['customer_id'], how='left')

    # Buy Month Mean 추가
    all_train_data = all_train_data.merge(train_buy_mean, on=['customer_id'],how='left')
    test_data = test_data.merge(test_buy_mean, on=['customer_id'],how='left')

    # December feature 추가
    all_train_data = all_train_data.merge(train_december, on=['customer_id'],how='left')
    test_data = test_data.merge(test_december, on=['customer_id'],how='left')

    # Over
    all_train_data = all_train_data.merge(train_over, on=['customer_id'],how='left')
    test_data = test_data.merge(test_over, on=['customer_id'],how='left')

    all_train_data['over_rate']=pd.to_numeric(all_train_data['over_count'])/(all_train_data['live_month']+1)
    all_train_data=all_train_data.drop(['over_count','live_month'],axis=1)

    test_data['over_rate']=pd.to_numeric(test_data['over_count'])/(test_data['live_month']+1)
    test_data=test_data.drop(['over_count','live_month'],axis=1)

    # Cohort feature 추가
    all_train_data = all_train_data.merge(train_cohort, on=['customer_id'],how='left')
    test_data = test_data.merge(test_cohort, on=['customer_id'],how='left')

    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, all_train_data['label'], features


if __name__ == '__main__':
    
    print('data_dir', data_dir)
