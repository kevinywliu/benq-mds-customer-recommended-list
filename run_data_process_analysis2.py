# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 22:41:01 2021

@author: howar
"""

# Utils
import plotly.express as px
import base64 
from sklearn.preprocessing import StandardScaler
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
import streamlit as st 
import numpy as np
import pandas as pd 
from stqdm import stqdm
import os
from xgboost import XGBClassifier
from util import get_dummies, detect_str_columns,model_testRF,results_summary_to_dataframe,plot_confusion_matrix,logistic_model,logistic_importance,logistic_conf,model_profit_fun,model_profit_newdata_fun
from util import profit_linechart, profit_linechart_all
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, auc, accuracy_score,classification_report
import os
import plotly.express as px

# ----設定繪圖-------
import matplotlib.pyplot as plt
import seaborn as sns 

@st.cache(suppress_st_warning=True)

    
def data_process_ml(data ='data', y='buy', UID = 'UID' ):
    
    
    # data = pd.read_csv('contract.csv')
    
    # 二 、 變數視覺化
    
    
    #  1. 使用 - 獨熱編碼
    str_columns = detect_str_columns(data)
    dataset = get_dummies(str_columns, data)
    st.write('轉換後資料模樣')
    st.dataframe(dataset.head(5))
    
     
    # ## 四 、 資料處理與轉換 - 切分資料集
     
    #  將X與y分割開來
     
    # 將 x 與 y 分割開來
    
    
    X =dataset.drop(columns=[y])
    y =dataset[y]
    
     
    # 切分資料集
    
    
    # 切分資料集
    
    '''
    切分成80％的訓練資料集
        - X_train : X訓練變數
        - y_train : y訓練變數
        
    切分成20％的測試資料集
        - X_test : X測試變數
        - y_test : y測試變數
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
    
     
    # 查看各自的維度
    
    
    # 來看看各自的維度
    print(X_train.shape)
    print(y_train.shape)
    
    print(X_test.shape)
    print(y_test.shape)
    
     
    # ## 五、資料處理與轉換 - 將UID 拿出來
     
    #  1. 保留UID
    
    
    # 保留UID  做推薦清單時需要將uid放回去 找到推薦人
    train_uid = X_train[UID]
    test_uid = X_test[UID]
    
     
    #  2. 刪除UID
    
    
    del X_train[UID]
    del X_test[UID]
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # 儲存模型
    return X_train, X_test, y_train, y_test, train_uid,test_uid,sc


# function
def client_list(model, y_test, X_test, test_uid, name):
    '''
    model:要放入的模型
    y_test：要驗證的資料集，如果沒有，請設定None
    X_test:要預測的資料集
    test_uid:這資料集的UID
    
    '''
    
    # prediction
    xgb_pred_prob = model.predict_proba(X_test)
    xgb_pred = model.predict(X_test)
    
    # 產出【顧客產品推薦名單】
    if y_test !=None:
        XGBClassifier_test_df=pd.DataFrame(y_test.values ,columns =['客戶對A商品【實際】購買狀態'])
        XGBClassifier_test_df['客戶對A商品【預測】購買機率'] = xgb_pred_prob[:,1]
        test_uid = test_uid.reset_index().drop(columns = ['index'])
        XGBClassifier_test_df = pd.concat([test_uid,XGBClassifier_test_df], axis = 1)
        XGBClassifier_test_df = XGBClassifier_test_df.sort_values('客戶對A商品【預測】購買機率', ascending = False)
        
    else:
        XGBClassifier_test_df= pd.DataFrame( xgb_pred_prob[:,1],columns =['客戶對A商品【預測】購買機率'])
        XGBClassifier_test_df = XGBClassifier_test_df.sort_values(['客戶對A商品【預測】購買機率'], ascending = False)
        test_uid = test_uid.reset_index().drop(columns = ['index'])
        XGBClassifier_test_df = pd.concat([test_uid,XGBClassifier_test_df], axis = 1)
        XGBClassifier_test_df = XGBClassifier_test_df.sort_values('客戶對A商品【預測】購買機率', ascending = False)
    
    
    # XGBClassifier_test_df.to_csv(name+'顧客產品推薦名單.csv',encoding = 'utf-8-sig')
    with st.beta_expander(label=name+ " 顧客產品推薦名單 ", expanded=True):
        st.write(name+'顧客產品推薦名單')
        st.dataframe(XGBClassifier_test_df)
        download = csv_downloader(XGBClassifier_test_df,
                                  filename=name+'顧客產品推薦名單')
        
    # return XGBClassifier_test_df,xgb_pred



import pickle
def download_model(model,filename):
    output_model = pickle.dumps(model)
    # output_model  = pickle.dump(model, open(filename, "wb"))
    b64 = base64.b64encode(output_model).decode()
    new_filename = filename+".pkl"
    st.markdown("#### Download 模型檔案 ###")
    href = f'<a href="data:file/output_model;base64,{b64}" download="{new_filename}">請點此下載模型檔案</a>'
    st.markdown(href, unsafe_allow_html=True)
    
def csv_downloader(data,filename):
	csvfile = data.to_csv()
	b64 = base64.b64encode(csvfile.encode("UTF-8-sig")).decode()
	new_filename = filename+"_{}_.csv".format(timestr)
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">請點此下載</a>'
	st.markdown(href,unsafe_allow_html=True)

# from joblib import dump,load
# dump(sc, 'std_scaler.bin', compress=True)
# sc=load('std_scaler.bin')

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    tick_marks  =np.array([tick_marks[0]-0.5, tick_marks[1]+0.5])
    plt.yticks(tick_marks, classes, rotation=1)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot(fig)
    
    
def model_fun(clf, X_train,y_train,X_test,y_test,
                    plot_name = 'logistic_regression') :
    
    # xgb_model.fit(X_train, y_train, verbose=True, eval_set=[(X_train, y_train), (X_test, y_test)])
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]

    y_test_df=pd.DataFrame(y_test)
    y_test_df[plot_name+'_pred'] = y_pred_prob
    
    y_test_df.columns = ['buy', plot_name+'_pred']
    
    
    #Confusion Matrix
    conf_logist = confusion_matrix(y_test, y_pred)
    
    
    with st.beta_expander(label=plot_name+ " Summary "):
          # 畫conf matrix
        plot_confusion_matrix(conf_logist, ['no','buy'],
                              title=plot_name+"Confusion Matrix plot", cmap=plt.cm.Reds)#, cmap=plt.cm.Reds
        
        # print(confusion_matrix(y_test, y_pred))
        st.write("Training Accuracy = {:.3f}".format(model.score(X_train, y_train)))
        st.write("Test Accuracy = {:.3f}".format(model.score(X_test, y_test)))
        
        # 顧客產品推薦名單
        y_test_df = y_test_df.sort_values([plot_name+'_pred'], ascending = False)
   
        # 下載csv
        # y_test_df.to_csv(plot_name+'顧客產品推薦名單.csv',encoding = 'utf-8-sig')
        st.write(plot_name+'顧客產品推薦名單')
        st.dataframe(y_test_df)
        download = csv_downloader(y_test_df,
                                  filename=plot_name+'顧客產品推薦名單')
        
        # 儲存模型
        st.write(plot_name+' Download 模型檔案')
        download_model(clf, plot_name)
    
def run_data_process_analysis():
    st.subheader("顧客CRM資料處理與分析")
    
    #----- side bar-----
    st.sidebar.subheader('請上傳資料')
    
    
    uploaded_file = st.file_uploader("請上傳您的顧客CRM資料分析CSV檔案，若無，則會使用範例資料集操作", type=["csv"])
    
    # 要上傳檔案
    if uploaded_file is not None:
        sales_data = pd.read_csv(uploaded_file)
        with st.beta_expander("上傳的資料表模樣"):
            st.dataframe(sales_data.iloc[0:10])
        
    else:
        st.write('等待上傳資料輸入~')
        sales_data = pd.read_csv('contract.csv')

        with st.beta_expander("範例資料表模樣"):
            st.dataframe(sales_data.iloc[0:10])
    
    
    
    
    st.subheader('請輸入目標變數')    
    y = st.text_input("請輸入目標變數", 'buy')
    
    st.subheader('請輸入本資料集的User id')   
    UID = st.text_input("請輸入本資料集的User id", 'UID')
    
    
    # Buy 比例
    st.text('不購買'+ str(round(sales_data[y].value_counts()[0]/len(sales_data) * 100,2))+ '% of the sales_dataset' )
    st.text('購買'+ str(round(sales_data[y].value_counts()[1]/len(sales_data) * 100,2))+ '% of the sales_dataset')
    
    # 看看 y變數的分佈圖
    sales_data['count'] = 1
    sales_data_count = sales_data.groupby(y, as_index=False)['count'].sum()
    fig = px.bar(sales_data_count , x="buy", y="count",
                  color='count', 
                 title ='purchase decision \n (0:no vs 1:yes )',
                 )
    
    st.plotly_chart(fig)
    del sales_data['count']
    #----- 正式頁面-----
    
    st.header('開始資料處理與分析...')
    X_train, X_test, y_train, y_test, train_uid,test_uid,sc = data_process_ml(data =sales_data, y=y, UID = UID)
    
    
    
    # 儲存模型
    st.write(' Download 標準化資料處理檔案')
    download_model(sc, '標準化資料處理檔案')
    
    with st.beta_expander('篩選random Forest ML模型參數', expanded=True):
        col1, col2 = st.beta_columns(2)
        with col1:
            n_estimators_rf = st.number_input('n_estimators', min_value=10, value=100)
            random_state_rf = st.number_input('random_state', min_value=0, value=0)          
        
        with col2:
            n_jobs = st.number_input('n_jobs', min_value=-1, value=3)          
            verbose= st.number_input('顯示', min_value=0, value=1)          
            
    
    
    with st.beta_expander('篩選XGBoost ML模型參數', expanded=True):
        col1, col2 = st.beta_columns(2)
        with col1:
            n_estimators = st.number_input('XGBoost_n_estimators', min_value=10, value=100)
            random_state = st.number_input('XGBoost_random_state', min_value=0, value=0)          
        
        with col2:
            nthread = st.number_input('nthread', min_value=-1, value=3)          
            learning_rate= st.number_input('XGBoost_learning_rate', min_value=0.001, value=0.5)          
                
    if_apply2 = st.button('Confirm to train model')
    if if_apply2:
        st.write('''  ## 訓練模型！'''  )
        # xgb setting
        xgb_model = XGBClassifier(n_estimators=n_estimators ,
                                  random_state = random_state,
                                  nthread = nthread , 
                                  learning_rate=learning_rate)
        # rf setting
        rf_model =RandomForestClassifier(n_estimators = n_estimators_rf, 
                                         random_state = random_state_rf,
                                         verbose=verbose, n_jobs=n_jobs)
        
        logistic_reg =LogisticRegression()
        
        # 訓練與預測
        model_fun(xgb_model, X_train,y_train,X_test,y_test,
                    plot_name = 'XGBoost')
        
        model_fun(rf_model, X_train,y_train,X_test,y_test,
                    plot_name = 'Random_Forest')
        
        
        model_fun(logistic_reg, X_train,y_train,X_test,y_test,
                    plot_name = 'Logistic_regression')
            

def prediction():
    
    
    #-----  bar-----
    st.subheader('請上傳您要預測顧客CRM推薦請的資料CSV檔案')
    
    uploaded_file = st.file_uploader("請上傳您要預測顧客CRM推薦請的資料CSV檔案，若無，則會使用範例資料集操作", type=["csv"])
    
    # 要上傳檔案
    if uploaded_file is not None:
        sales_data = pd.read_csv(uploaded_file)
        with st.beta_expander("上傳的資料表模樣"):
            st.dataframe(sales_data.iloc[0:10])
        
    else:
        st.write('等待上傳資料輸入~')
        sales_data = pd.read_csv('contract_newdata.csv')

        with st.beta_expander("範例資料表模樣"):
            st.dataframe(sales_data.iloc[0:10])
    
    #-----  bar-----
    st.subheader('請上傳模型訓練檔案')
    
    #uploaded_file = st.file_uploader("請上傳之前dat模型訓練檔案", type=["dat"])
    uploaded_file = st.file_uploader("請上傳之前pkl模型訓練檔案")
    
    # 要上傳檔案
    if uploaded_file is not None:
        # model_xgb = pd.read_csv(uploaded_file)
        model_xgb = pickle.load(open(uploaded_file, "rb"))
        
    else:
        st.write('等待上傳資料輸入~')
        # sales_data = pd.read_csv('XGB_model.pkl')
        model_xgb = pickle.load(open("XGBoost.pkl", "rb"))
    
    
    #-----  bar-----
    st.subheader('請上傳標準化資料處理檔案')
    
    #uploaded_file = st.file_uploader("請上傳之前標準化資料處理檔案", type=["dat"])
    uploaded_file = st.file_uploader("請上傳之前標準化資料處理檔案")
    
    # 要上傳檔案
    if uploaded_file is not None:
        # model_xgb = pd.read_csv(uploaded_file)
        sc = pickle.load(open(uploaded_file, "rb"))
        
    else:
        st.write('等待上傳資料輸入~')
        # sales_data = pd.read_csv('XGB_model.pkl')
        sc = pickle.load(open("標準化資料處理檔案.pkl", "rb"))
    
    
    st.subheader('請輸入本資料集的User id')   
    UID = st.text_input("請輸入本資料集的User id", 'UID')
    
    st.subheader('請輸入本資料集的名稱')   
    name = st.text_input("請輸入本資料集的名稱",'test')
    
    
    newdata_uid = sales_data[UID]
    del sales_data[UID]
    sales_data = sales_data.values
    
    # model_xgb.get_booster().feature_names = sales_data.columns.tolist()
    
    sales_data = sc.transform(sales_data)
    
    
    client_list(model = model_xgb,
                y_test=None, 
                X_test=sales_data, 
                test_uid=newdata_uid, 
                name = name)
    
    
    