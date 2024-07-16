# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:42:13 2024

@author: leehj
"""

# # 환경 설정

# In[1]:


# 구글 드라이브 마운트
"""
from google.colab import drive
drive.mount('/gdrive')
"""

# In[2]:


# 라이브러리 설정
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import matplotlib
import missingno # pip install missingno

print(np.__version__)         # 1.25.2
print(pd.__version__)         # 2.1.0
print(sns.__version__)        # 0.12.2
print(sklearn.__version__)    # 1.3.1
print(matplotlib.__version__) # 3.7.2
print(missingno.__version__)  # 0.5.2

#%%
## 데이터셋 불러오기


# 데이콘 사이트에서 다운로드한 CSV파일을 읽어오기
# drive_path = "/gdrive/My Drive/"
drive_path = "./titanic/"

train = pd.read_csv(drive_path + "train.csv")
test = pd.read_csv(drive_path + "test.csv")
submission = pd.read_csv(drive_path + "submission.csv")

print(train.shape, test.shape, submission.shape)


#%%

# In[4]:


# train 데이터프레임 내용 확인
train.head(3)


# In[5]:


# test 데이터프레임 내용 확인
test.head(2)


# In[6]:


# submission 제출파일 양식 확인
submission.head()


# # 데이터 살펴보기

# ### 데이터 구조

# In[7]:


# train 데이터프레임 개요 정보
train.info()


# In[8]:


# train 데이터프레임 통계정보
train.describe(include='all')


# ### 결측값 확인

# In[9]:


# 결측값 분포
import missingno as msno
msno.bar(train, figsize=(10, 5), color=(0.7, 0.2, 0.2))
plt.show()


# In[10]:


msno.matrix(test, figsize=(10, 5), color=(0.7, 0.2, 0.2))
plt.show()

#%%
# 히스토그램

sns.histplot(data=train, x="Age")
sns.histplot(data=train, x="Age", bins=10)
sns.histplot(data=train, x="Age", hue='Sex')
sns.histplot(data=train, x="Age", hue='Sex', multiple="stack")
sns.histplot(data=train, x="Age", hue='Survived')
sns.histplot(data=train, x="Age", hue='Survived', multiple="stack")

#%%

# 학습데이터 + 테스트 데이터
titanic = pd.concat([train, test], axis=0)
titanic.reset_index(inplace=True)
# titanic.dropna(subset=['Age'], axis=0, inplace=True)
titanic2 = titanic.dropna(subset=['Age'], axis=0)

# 커널밀도추정
sns.kdeplot(data=titanic, x='Age')


# In[11]:

# 상관관계
train_corr_df = train.loc[:, ['Pclass','Age','SibSp','Parch','Fare','Survived']]
train_corr = train_corr_df.corr()

#%%
# 숫자형 변수 간의 상관관계를 계산하여 히트맵 그리기
plt.figure(figsize=(8, 8))
sns.set(font_scale=0.8)
# sns.heatmap(train.corr(), annot=True, cbar=True);
sns.heatmap(train_corr, annot=True, cbar=True);
plt.show()


# ### 데이터셋 결합 

# In[12]:


# 타이타닉 전체 데이터셋 준비
train['TrainSplit'] = 'Train'
test['TrainSplit'] = 'Test'
data = pd.concat([train, test], axis=0)
print(data.shape)


# # 베이스라인 모델 학습
# - 숫자형 데이터만 사용

# ### 데이터 전처리

# In[13]:


# 숫자형 피처 추출
data_num = data.loc[:, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]

# 결측값 대체
data_num['Age'] = data_num['Age'].fillna(data_num['Age'].mean())
data_num['Fare'] = data_num['Fare'].fillna(data_num['Fare'].mode()[0])

# 학습용 데이터와 예측 대상인 테스트 데이터 구분
selected_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

X_train = data_num.loc[data['TrainSplit']=='Train', selected_features] 
y_train = data_num.loc[data['TrainSplit']=='Train', 'Survived']

X_test = data_num.loc[data['TrainSplit']=='Test', selected_features]

print("Train 데이터셋 크기: ", X_train.shape, y_train.shape)
print("Test 데이터셋 크기: ", X_test.shape)


# ### 교차 검증 

# In[14]:


# 훈련 - 검증 데이터 분할
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val =  train_test_split(X_train, y_train, test_size=0.2, 
                                             shuffle=True, random_state=20)

# 로지스틱 회귀 모델
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_tr, y_tr)
y_val_pred = lr_model.predict(X_val)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_val, y_val_pred), annot=True, cbar=False, square=True)
plt.show()


# In[15]:

# confusion_matrix
"""
실제값 -> F: TN FP
          T: FN TP
             F  T  <- 예측값
"""
print("confusion_matrix:")
cm = confusion_matrix(y_val, y_val_pred)
FP = cm[0,1]
TP = cm[1,1]
FN = cm[1,0]

# pc = cm[1,1] / (cm[0,1] + cm[1,1])
pc = TP / (TP + FP)
print(cm)
print("precision(정밀도): ", round(pc, 4))

# rc = cm[1,1] / (cm[1,0] + cm[1,1])
rc = TP / (TP + FN)
print("recall(재현율): ", round(rc,4))

F1 = 2 * (pc * rc) / (pc + rc)
print("F1 score(재현율): ", round(F1,4))


#%%
# 평가 지표
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score 
print("Accuracy: %.4f" % accuracy_score(y_val, y_val_pred))
print("Precision: %.4f" % precision_score(y_val, y_val_pred))
print("Recall: %.4f" % recall_score(y_val, y_val_pred))
print("F1: %.4f" % f1_score(y_val, y_val_pred))
print("AUC: %.4f" % roc_auc_score(y_val, y_val_pred))


# ### 제출 파일 만들기

# In[16]:


# test 데이터에 대한 예측값 정리
y_test_pred = lr_model.predict(X_test)

# 제출양식에 맞게 정리
submission['Survived'] = y_test_pred.astype(int)

# 제출파일 저장 
submission_filepath = drive_path + 'baseline_num_lr_submission_001.csv'    
submission.to_csv(submission_filepath, index=False)
submission.head(5)


# # 피처 엔지니어링
# - 문자열, 범주형 변수를 포함
# - 데이터 탐색(EDA) 및 전처리(pre-processing) 포함

# ### Survived : 생존 여부

# In[17]:


# 타겟 레이블의 분포 확인
train['Survived'].value_counts(dropna=False)


# In[18]:


# 객실 등급별 분포 확인
sns.countplot(x='Survived', data=data[data['TrainSplit']=='Train'])
plt.show()


# ### Pclass : 객실 등급

# In[ ]:


# train - test 데이터 분포
sns.countplot(x='Pclass', hue='TrainSplit', data=data)
plt.show()


# In[ ]:


# Pclass별 Survived 여부
sns.countplot(x='Pclass', hue='Survived', data=data[data['TrainSplit']=='Train'])
plt.show()


# In[ ]:


# Pclass별 Fare 객실요금의 중간값 비교
sns.barplot(x='Pclass', y='Fare', hue='Survived', 
            data=data[data['TrainSplit']=='Train'], estimator=np.median)
plt.show()


# ### Sex : 성별

# In[ ]:


# histplot 함수 - dodge 옵션
sns.histplot(x='Sex', hue='Survived', multiple='dodge', 
             data=data[data['TrainSplit']=='Train'])
plt.show()


# In[ ]:


# histplot 함수 - stack 옵션
sns.histplot(x='Sex', hue='Survived', multiple='stack', 
             data=data[data['TrainSplit']=='Train'])
plt.show()


# In[ ]:


# histplot 함수 - fill 옵션
sns.histplot(x='Sex', hue='Survived', multiple='fill', 
             data=data[data['TrainSplit']=='Train'])
plt.show()


# In[ ]:


# 레이블 인코딩 (female: 0, male: 1)
data.loc[data['Sex']=='female', 'Sex'] = 0
data.loc[data['Sex']=='male', 'Sex'] = 1
data['Sex'] = data['Sex'].astype(int)

# 성별 분포 확인
data['Sex'].value_counts(dropna=False)


# ### Name : 이름
# - 문자열 피처 다루기 : Name(이름)에서 Mr. Dona 등 title 정보를 추출 (신분을 나타내는 정보)

# In[ ]:


data['Name'].unique()


# In[ ]:


title_name = data['Name'].str.split(", ", expand=True)[1]
title_name


# In[ ]:


title = title_name.str.split(".", expand=True)[0]
title.value_counts(dropna=False)


# In[ ]:


title = title.replace(['Ms'], 'Miss')
title = title.replace(['Mlle', 'the Countess', 'Lady', 'Don', 'Dona', 'Mme', 'Sir', 'Jonkheer'], 'Noble')
title = title.replace(['Col', 'Major', 'Capt'], 'Officer')
title = title.replace(['Dr', 'Rev'], 'Priest')
data['Title'] = np.array(title)
data['Title'].value_counts(dropna=False)


# In[ ]:


# 승객 나이와 생존 여부와의 관계
sns.violinplot(x='Title', y='Age', hue='Survived', data=data,  split=True)
plt.show()


# In[ ]:


# Name 열 삭제
data = data.drop('Name', axis=1)
data.columns


# ### Age : 나이

# In[ ]:


#결측값 확인 및 대체
for title in data['Title'].unique():
    # 결측값 개수 확인
    print("%s 결측값 개수: " % title, data.loc[data['Title']==title, 'Age'].isnull().sum())  
    # 각 Title의 중앙값으로 대체  
    age_med = data.loc[data['Title']==title, 'Age'].median()
    data.loc[data['Title']==title, 'Age'] = data.loc[data['Title']==title, 'Age'].fillna(age_med)

# 결측값 처리 여부 확인
print("\n")
print("Age 열의 결측값 개수: ", data['Age'].isnull().sum())


# In[ ]:


# Age 분포
sns.displot(x='Age', kind='hist', hue='Survived', 
            data=data[data['TrainSplit']=='Train'])
plt.show()


# In[ ]:


# Binning - 구간 나누기
bins = [0, 4, 8, 12, 16, 32, 36, 48, 56, 64, 100]
labels = ['Infant', 'Child1', 'Child2', 'Youth1', 'Youth2', 'Adult1', 'Adult2','MIddle Aged', 'Senior', 'Elderly']
data['AgeBin'] = pd.cut(data['Age'], bins=bins, labels=labels)

# Age_bin (나이 구간)에 따른 생존율 비교
sns.countplot(x = 'AgeBin', hue = 'Survived', 
              data=data[data['TrainSplit']=='Train'])
plt.xticks(rotation=45)
plt.show()


# ### SibSp: 형제자매/배우자

# In[ ]:


# 형제자매/배우자 수와 승객 나이 및 생존율 관계
sns.boxplot(x='SibSp', y='Age', hue='Survived', 
            data=data[data['TrainSplit']=='Train'])
plt.show()


# ### Parch:부모자식

# In[ ]:


# 부모자식 수에 따른 승객 나이 및 생존율 관계
sns.boxplot(x='Parch', y='Age', hue='Survived', 
            data=data[data['TrainSplit']=='Train'])
plt.show()


# * 피처 교합 (feature interaction)

# In[ ]:


# 가족 구성원의 수
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# 가족 구성원의 수와 생존율 관계
sns.barplot(x='FamilySize', y='Survived', hue='Pclass', estimator=np.mean,
            data=data[data['TrainSplit']=='Train'])
plt.show()


# ### Fare : 요금

# In[ ]:


# 결측값 확인
data.loc[data['Fare'].isnull(), :]


# In[ ]:


# 3등석 요금의 평균값을 가지고 결측값 대체
p3_fare_mean = data.loc[data['Pclass']==3, 'Fare'].mean()
print(p3_fare_mean)
data['Fare']= data['Fare'].fillna(p3_fare_mean)
data.loc[data['PassengerId']==1044, :'Fare']


# In[ ]:


# Fare 분포
sns.displot(x='Fare', kind='kde', hue='Survived', 
            data=data[data['TrainSplit']=='Train'])
plt.show()


# In[ ]:


# log 변환
data['FareLog'] = np.log1p(data['Fare'])

# FareLog 분포
sns.displot(x='FareLog', kind='hist', hue='Survived', 
            data=data[data['TrainSplit']=='Train'])
plt.show()


# In[ ]:


# 객실 등급 별 객실 요금 분포와 생존율
sns.stripplot(x='Pclass', y='FareLog', hue='Survived', 
              data=data[data['TrainSplit']=='Train'])
plt.show()


# ### Embarked : 탑승 항구

# In[ ]:


# 결측값 확인
data.loc[data['Embarked'].isnull(), :]


# In[ ]:


# 최빈값을 사용하여 결측값 처리
print("Embarked 열의 최빈값:", data['Embarked'].mode()[0])
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Embarked'].value_counts(dropna=False)


# In[ ]:


# 탑승 항구별 생존율 비교
sns.catplot(x='Embarked',  y='Survived', kind='point',
               data=data[data['TrainSplit']=='Train'])
plt.show()


# ### Cabin : 객실 구역

# In[ ]:


# 고유값 확인
data['Cabin'].unique()


# In[ ]:


# 첫번째 알파벳 이니셜 추출
data['Cabin'].str.slice(0, 1).value_counts(dropna=False)


# In[ ]:


# 알파벳 이니셜로 대체, 결측값은 'U'로 입력
data['Cabin'] = data['Cabin'].str.slice(0, 1)
data['Cabin'] = data['Cabin'].fillna('U')

# Cabin 구역별 생존율 비교
sns.catplot(x='Cabin',  y='Survived', kind='bar',
               data=data[data['TrainSplit']=='Train'])
plt.show()


# ### Ticket : 티켓 정보

# In[ ]:


# 고유값 확인
data['Ticket'].value_counts(dropna=False)


# In[ ]:


# 문자열 정리 - 알파벳 추출
data['Ticket'] = data['Ticket'].str.replace(".","").str.replace("/","")
data['Ticket'] = data['Ticket'].str.strip().str.split(' ').str[0]
data['Ticket'].value_counts(dropna=False)


# In[ ]:


# 문자열이 숫자인 경우에는 "NUM"으로 대체
data.loc[data['Ticket'].str.isdigit(), 'Ticket'] = 'NUM'
data['Ticket'].value_counts(dropna=False)[:10]

# Ticket 번호별 생존율 비교
sns.catplot(x='Ticket',  y='Survived', kind='bar',
               data=data[data['TrainSplit']=='Train'])
plt.xticks(rotation=90)
plt.show()


# # 데이터 전처리

# ### 레이블 인코딩

# In[ ]:


# Label Encoding
from sklearn.preprocessing import LabelEncoder
for col in ['Title', 'AgeBin']:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])

data.loc[:, ['Title', 'AgeBin']].head()


# ### 원핫인코딩

# In[ ]:


# 범주형 변수로 변환 및 원핫인코딩
onehot_prefix = []
for col in ['Embarked', 'Cabin', 'Ticket']:
    data[col] = data[col].astype('category')
    data = pd.get_dummies(data, columns = [col], prefix=col[:3], drop_first=True)
    onehot_prefix.append(col[:3])

data.loc[:,[col for col in data.columns if col[:3] in onehot_prefix]].head()


# ### 피처 스케일링

# In[ ]:


# 피처 스케일링
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

# 스케일링 처리할 피처 선택 - TrainSplit 등 일부 열을 제외
scaled_cols = [col for col in data.loc[:, 'Pclass':].columns if col!='TrainSplit']

data_scaled = data.loc[:, scaled_cols]
data_scaled = scaler.fit_transform(data_scaled)

# 스케일링 변환된 값을 데이터프레임에 반영
data.loc[:, scaled_cols] = data_scaled[:, :]
data.head()


# # 모델 학습

# ### 피처 선택

# In[ ]:


selected_features = ['Pclass', 'Sex', 'SibSp', 'Parch',
       'Title', 'AgeBin', 'FamilySize', 'FareLog', 
       'Emb_Q', 'Emb_S', 'Cab_B', 'Cab_C', 'Cab_D', 'Cab_E', 'Cab_F', 'Cab_G',
       'Cab_T', 'Cab_U', 'Tic_A4', 'Tic_A5', 'Tic_AQ3', 'Tic_AQ4', 'Tic_AS',
       'Tic_C', 'Tic_CA', 'Tic_CASOTON', 'Tic_FC', 'Tic_FCC', 'Tic_Fa',
       'Tic_LINE', 'Tic_LP', 'Tic_NUM', 'Tic_PC', 'Tic_PP', 'Tic_PPP',
       'Tic_SC', 'Tic_SCA3', 'Tic_SCA4', 'Tic_SCAH', 'Tic_SCOW', 'Tic_SCPARIS',
       'Tic_SCParis', 'Tic_SOC', 'Tic_SOP', 'Tic_SOPP', 'Tic_SOTONO2',
       'Tic_SOTONOQ', 'Tic_SP', 'Tic_STONO', 'Tic_STONO2', 'Tic_STONOQ',
       'Tic_SWPP', 'Tic_WC', 'Tic_WEP']

len(selected_features)


# ### Train-Test 데이터셋 분할

# In[ ]:


# 학습용 데이터와 예측 대상인 테스트 데이터 구분
y_train = data.loc[data['TrainSplit']=='Train', 'Survived']
X_train = data.loc[data['TrainSplit']=='Train', selected_features] 
X_test = data.loc[data['TrainSplit']=='Test', selected_features]
print("Train 데이터셋 크기: ", X_train.shape, y_train.shape)
print("Test 데이터셋 크기: ", X_test.shape)


# ### 교차 검증 

# In[ ]:


# 훈련 - 검증 데이터 분할
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val =  train_test_split(X_train, y_train, test_size=0.2, 
                                             shuffle=True, random_state=20)
print("훈련 데이터셋 크기: ", X_tr.shape, y_tr.shape)
print("검증 데이터셋 크기: ", X_val.shape, y_val.shape)


# In[ ]:


# 로지스틱 회귀 모델
lr_model = LogisticRegression()
lr_model.fit(X_tr, y_tr)

y_tr_pred = lr_model.predict(X_tr)
print("훈련 Accuracy: %.4f" % accuracy_score(y_tr, y_tr_pred))
print("훈련 AUC: %.4f" % roc_auc_score(y_tr, y_tr_pred))

y_val_pred = lr_model.predict(X_val)
print("검증 Accuracy: %.4f" % accuracy_score(y_val, y_val_pred))
print("검증 AUC: %.4f" % roc_auc_score(y_val, y_val_pred))


# In[ ]:


# test 데이터 예측 및 제출파일 저장
y_test_pred = lr_model.predict(X_test)
submission['Survived'] = y_test_pred.astype(int)
submission_filepath = drive_path + 'baseline_lr_submission_002.csv'    
submission.to_csv(submission_filepath, index=False)


# # 모델 성능 개선하기

# ### 다른 모델 적용해 보기

# In[ ]:


# 랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=2020)
# cross_val_score 함수
from sklearn.model_selection import cross_val_score
auc_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='roc_auc')
print("개별 Fold의 AUC 점수: ", np.round(auc_scores, 4))
print("평균 AUC 점수: ", np.round(np.mean(auc_scores), 4))  


# In[ ]:


# 제출 파일
rf_model.fit(X_train, y_train)
y_test_pred = rf_model.predict(X_test)
submission['Survived'] = y_test_pred.astype(int)
submission_filepath = drive_path + 'baseline_rf_submission_001.csv'    
submission.to_csv(submission_filepath, index=False)


# ### 피처 중요도

# In[ ]:


# tree 계열 알고리즘의 feature importance 그래프
def plot_importance(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    feature_names = [features[i] for i in indices]
    feature_imp =  importances[indices]

    plt.figure(figsize=(10, 12))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)),feature_imp, align='center')
    plt.yticks(range(len(indices)), feature_names)
    plt.xlabel('Relative Importance')

    print('피처: ', list(reversed(feature_names)))
    print('중요도: ', list(reversed(feature_imp)))

    return list(reversed(feature_names)), list(reversed(feature_imp))


# In[ ]:


# 랜덤 포레스트 모델의 피처 중요도
imp_features, imp_scores = plot_importance(rf_model, selected_features)


# In[ ]:


# 상위 10개 피처만 선택
selected_features = imp_features[:10]      
y_train = data.loc[data['TrainSplit']=='Train', 'Survived']
X_train = data.loc[data['TrainSplit']=='Train', selected_features] 
X_test = data.loc[data['TrainSplit']=='Test', selected_features]
print("Train 데이터셋 크기: ", X_train.shape, y_train.shape)
print("Test 데이터셋 크기: ", X_test.shape)


# In[ ]:


# 랜덤 포레스트
rf_model = RandomForestClassifier(random_state=2020)
auc_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc')
print("개별 Fold의 AUC 점수: ", np.round(auc_scores, 4))
print("평균 AUC 점수: ", np.round(np.mean(auc_scores), 4))  
rf_model.fit(X_train, y_train)
y_test_pred = rf_model.predict(X_test)
submission['Survived'] = y_test_pred.astype(int)
submission_filepath = drive_path + 'baseline_rf_submission_002.csv'    
submission.to_csv(submission_filepath, index=False)


# In[ ]:


# XGBoost
from xgboost import XGBClassifier
xgb_model = XGBClassifier(max_depth=3, random_state=2020)
auc_scores = cross_val_score(xgb_model, X_train, y_train, cv=3, scoring='roc_auc')
print("개별 Fold의 AUC 점수: ", np.round(auc_scores, 4))
print("평균 AUC 점수: ", np.round(np.mean(auc_scores), 4))  
xgb_model.fit(X_train, y_train)
y_test_pred = xgb_model.predict(X_test)
submission['Survived'] = y_test_pred.astype(int)
submission_filepath = drive_path + 'baseline_xgb_submission_001.csv'    
submission.to_csv(submission_filepath, index=False)


# ### 분류 확률값으로 제출파일 만들기

# In[ ]:


# 확률값 예측
y_xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
y_rf_proba = rf_model.predict_proba(X_test)[:, 1]

# 앙상블 기법
y_proba = (y_xgb_proba + y_rf_proba) / 2
submission['Survived'] = y_proba
submission_filepath = drive_path + 'baseline_proba_submission_001.csv'    
submission.to_csv(submission_filepath, index=False)