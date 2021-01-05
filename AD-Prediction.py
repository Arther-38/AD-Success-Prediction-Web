# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 16:19:49 2021

@author: user
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
import base64
wkDir = "c:/Users/user/OneDrive/桌面/Dataset";   os.chdir(wkDir)
train=pd.read_csv('Train.csv')
test=pd.read_csv('Test.csv')
print(train)

#%% Data Visualize
def groupby(data,feat1,feat2):
    result=data.groupby([feat1,feat2]).size().unstack().plot(kind='bar',stacked=False)
    return result

def violin(data,feat1,feat2):
    return sns.violinplot(x=feat1, y=feat2, data=data)

def corr(data,feat1,feat2):
    corr = data[[feat1,feat2]].corr()
    return sns.heatmap(corr,annot=True,cmap='BuPu')

#%% -------------------------------------- Data Preprocessing ------------------------------
train=train.rename(columns={'average_runtime(minutes_per_week)':'ave_runtime'})
test=test.rename(columns={'average_runtime(minutes_per_week)':'ave_runtime'})
train['netgain']=np.where(train['netgain']=='False','FALSE',train['netgain'])
train['netgain']=np.where(train['netgain']=='True','TRUE',train['netgain'])
#%%
train['netgain']=np.where(train['netgain']=='True','TRUE',train['netgain'])
train['netgain']=np.where(train['netgain']=='False','FALSE',train['netgain'])
#%%
print(train['netgain'])
#%%
train1=train.copy()
test1=test.copy()
train1= train1.drop(['id'],axis=1)
test1 = test1.drop(['id'],axis=1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train1['netgain']=le.fit_transform(train1['netgain'])
#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
X=train1.drop(columns=['netgain'])
X = pd.get_dummies(X,columns=['relationship_status','industry','genre','targeted_sex','airtime','airlocation','expensive','money_back_guarantee'])
print(X)
#%%
y=train1.netgain
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print('X_train shape= ',X_train.shape)
print('X_test shape= ',X_test.shape)
#%%
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression(solver='liblinear')
logreg=logreg.fit(X_train,y_train)
print(y_train)
#%%

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
#y_test,X_test,logreg
def ROC_CURVE(data1,data2,model):

    logit_roc_auc = roc_auc_score(data2, model.predict(data1))
    fpr, tpr, thresholds = roc_curve(data2, model.predict_proba(data1)[:,1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
#%%
#print(ROC_CURVE(X_test,y_test,logreg))
#%%

from sklearn.metrics import confusion_matrix, classification_report
def evaluate_model(model, dataX, dataY):
    
    model_acc = model.score(dataX, dataY)
    print("Test Accuracy: {:.2f}%".format(model_acc * 100))
    
    y_true = np.array(dataY)
    y_pred = model.predict(dataX)
    
    cm = confusion_matrix(y_true, y_pred)
    clr = classification_report(y_true, y_pred, target_names=["NO NETGAIN", "GET NETGAIN"])
    
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
    plt.savefig('pic1.png')
    plt.xticks(np.arange(2) + 0.5, ["NO NETGAIN", "GET NETGAIN"])
    plt.yticks(np.arange(2) + 0.5, ["NO NETGAIN", "GET NETGAIN"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

y_true = np.array(y_test)
y_pred = logreg.predict(X_test)

#print("Classification Report:\n----------------------\n", clr)
#clr = classification_report(y_true, y_pred, target_names=["NO NETGAIN", "GET NETGAIN"])
#%%
#evaluate_model(logreg, X_test, y_test)
#%%

test1 = pd.get_dummies(test1,columns=['relationship_status','industry','genre','targeted_sex','airtime','airlocation','expensive','money_back_guarantee'])
#%%
print(test1)
#%%

missing_cols = set( X.columns ) - set( test1.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    test1[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
test1 = test1[X.columns]

test_pred = logreg.predict(test1)

print(test_pred[0:100])
#%%
import streamlit as st

data_select=['Selection','資料相關性繪圖','模型預測','預測結果']
model_select=['ROC曲線','結果分類報告','混淆矩陣']
category_type=['sequence','non-sequence','half']
nonfeature=['relationship_status','industry','genre','targeted_sex','airtime','airlocation','expensive','money_back_guarantee','netgain']
seqfeature=['ave_runtime','ratings']
st.title('AD-Success Prediction:')
st.subheader('訓練資料集')
st.write('Dataset Source: https://www.kaggle.com/chetanambi/predict-ad-success?select=Train.csv')
st.write('ad-train data: ')
st.write(train.drop(columns=['netgain'])[0:10])
st.write('ad-train- target: ')
st.write(train['netgain'][0:10])


st.sidebar.title('Welcome to my AI Website!')
#choice = st.sidebar.selectbox('繪圖看資料間的關係: ')
select= st.sidebar.selectbox('請選擇...',data_select)

st.sidebar.header('功能:')

if select=='資料相關性繪圖':
    type=st.sidebar.selectbox('有序/無序', category_type)
           
    if type == 'sequence':
        nameX1=st.sidebar.selectbox('請選擇一有序繪圖變量', seqfeature)
        nameX2=st.sidebar.selectbox('請選擇二有序繪圖變量', seqfeature)
        if st.sidebar.checkbox("開始繪圖 --"): 
            corr(train,nameX1,nameX2)
            st.pyplot()           
            
    elif type == 'non-sequence':
        nameX1=st.sidebar.selectbox('請選擇一無序繪圖變量', nonfeature)
        nameX2=st.sidebar.selectbox('請選擇二無序繪圖變量', nonfeature)
        if st.sidebar.checkbox("開始繪圖 --"):   
            groupby(train,nameX1,nameX2)     
            st.pyplot()
   
    elif type == 'half':
        nameX1=st.sidebar.selectbox('請選擇一無序繪圖變量', nonfeature)
        nameX2=st.sidebar.selectbox('請選擇二有序繪圖變量', seqfeature)
        if st.sidebar.checkbox("開始繪圖 --"):
            violin(train,nameX1,nameX2)   
            st.pyplot()
            
elif select =='模型預測':
    st.success('無序資料進行獨熱編碼, 有序資料進行標準化: ')
    st.write(X[0:10])
    st.write('shape:')
    st.json({'row':X.shape[0],'col':X.shape[1],'test_size':0.33})
    choose_model=['logistic Regression']
    model=st.sidebar.selectbox('有序/無序',choose_model)
    
    if model == 'logistic Regression':
        func=st.sidebar.selectbox('顯示: ',model_select)
        
        if func == 'ROC曲線':
            if st.sidebar.checkbox("開始繪圖 --"):
                ROC_CURVE(X_test,y_test,logreg)
                st.pyplot()
        elif func == '結果分類報告' :
            st.text('Model Report:\n ' + classification_report(y_true, y_pred,target_names=["NO NETGAIN", "GET NETGAIN"]))
        elif func == '混淆矩陣' :
            if st.sidebar.checkbox("開始繪圖 --"):
                evaluate_model(logreg, X_test, y_test)
                st.pyplot()
    
elif select =='預測結果':
    
        st.title('Prediction Result: ')
        relation=st.selectbox('Marriage',train['relationship_status'].unique())
        industry=st.selectbox('industry',train['industry'].unique())
        genre=st.selectbox('genre',train['genre'].unique())
        sex=st.selectbox('sex',train['targeted_sex'].unique())
        ave_runtime = st.text_input('average runtime (input time 0~99 (min))',40)
        airtime = st.selectbox('airtime',train['airtime'].unique())        
        airlocation=st.selectbox('airlocation',train['airlocation'].unique())
        ratings=st.text_input('ratings (input ratings 0~1)',0.02)
        expensive=st.selectbox('expensive',train['expensive'].unique())    
        money_back_guarantee=st.selectbox('money_back_guarantee',train['money_back_guarantee'].unique())
        result=[(relation,industry,genre,sex,ave_runtime,airtime,airlocation,ratings,expensive,money_back_guarantee)]
        st.subheader('Data Show')
        col1=train.drop(columns=['id','netgain']).columns
        new=pd.DataFrame(result,columns=col1)
            
        st.write(result)
        st.write(new)
        new=pd.get_dummies(new,columns=['relationship_status','industry','genre','targeted_sex','airtime','airlocation','expensive','money_back_guarantee'])
        hh=train.copy()
        hh=hh.drop(columns=['id','netgain'])
        print(hh)
        hh=pd.get_dummies(hh,columns=['relationship_status','industry','genre','targeted_sex','airtime','airlocation','expensive','money_back_guarantee'])
        X.columns=hh.columns
        missing_cols = set( X.columns ) - set( new.columns )
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            new[c] = 0
        # Ensure the order of column in the test set is in the same order than in train set
        new = new[X.columns]
        st.write(new)
        
        #print(new)

        final=logreg.predict(new)
        
        if final==0:
            final='False'
        else:
            final='True'
        if st.button('顯示結果:'):
            st.success(final)
        
        st.write('下載結果:')
        if st.button('download button'):
            dfprediction=pd.DataFrame ({'netgain':test_pred},columns=['netgain'])
            dfprediction[dfprediction['netgain']==0]='False'
            dfprediction[dfprediction['netgain']==1]='True'
            print(dfprediction)
            
            test_result=pd.concat([test,dfprediction],axis=1)
            
            from io import BytesIO
            def to_excel(df):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                df.to_excel(writer, sheet_name='Sheet1')
                writer.save()
                processed_data = output.getvalue()
                return processed_data
            
            def get_table_download_link(df):
                val = to_excel(df)
                b64 = base64.b64encode(val)  # val looks like b'...'
                return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download csv file</a>'
            st.markdown(get_table_download_link(test_result), unsafe_allow_html=True)

            print(test_result)
            output = test_result.to_csv('prediction_result.csv', index=False)

            st.write(test_result[0:20])
#%%

dfprediction=pd.DataFrame ({'netgain':test_pred},columns=['netgain'])
dfprediction[dfprediction['netgain']==0]='False'
dfprediction[dfprediction['netgain']==1]='True'
print(dfprediction)

test_result=pd.concat([test,dfprediction],axis=1)
print(test_result)
output = test_result.to_csv('prediction_result.csv', index=False)

#print(train1['netgain'].value_counts())
