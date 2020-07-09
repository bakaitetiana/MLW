#!/usr/bin/env python
# coding: utf-8

# In[2]:


#to access kaggle datasets
get_ipython().system('pip install kaggle')
#Math operations
get_ipython().system('pip install numpy==1.15.0')
#Machine learning
get_ipython().system('pip install catboost')


# In[3]:


#data preprocessing
import pandas as pd
#math operations
import numpy as np
#machine learning
from catboost import CatBoostRegressor, Pool
#data scaling
from sklearn.preprocessing import StandardScaler
#hyperparameter optimization
from sklearn.model_selection import GridSearchCV
#support vector machine model
from sklearn.svm import NuSVR, SVR
#kernel ridge model
from sklearn.kernel_ridge import KernelRidge
#data visualization
import matplotlib.pyplot as plt
from matplotlib import cm
import math


# In[119]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[5]:


#Extract training data into a dataframe for further manipulation
#train = pd.read_csv('C:\\Users\\talal\\Desktop\\WUT\\S3\\Machine Learning\\LANL-Earthquake-Prediction\\train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})


# In[222]:


train_reduced = train.values[::100]


# In[223]:


print(len(train_reduced))


# In[226]:


import csv
with open('train_reduced.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(train_reduced)
writeFile.close()


# In[57]:


import pickle
#save to file
#with open('train_ad_sample_df', 'wb') as f:
    #pickle.dump(train_ad_sample_df, f)
#with open('train_ttf_sample_df', 'wb') as f:
    #pickle.dump(train_ttf_sample_df, f)
#read from file
with open('train_ttf_sample_df', 'rb') as f:
    train_ttf_sample_df = pickle.load(f)
with open('train_ttf_sample_df', 'rb') as f:
    train_ttf_sample_df = pickle.load(f)


# In[59]:


#visualize 1% of samples data, first 100 datapoints
#train_ad_sample_df = train['acoustic_data'].values[::100]
#train_ttf_sample_df = train['time_to_failure'].values[::100]

#function for plotting based on both features
def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% sampled data"):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title(title)
    plt.plot(train_ad_sample_df, color='r')
    ax1.set_ylabel('acoustic data', color='r')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(train_ttf_sample_df, color='b')
    ax2.set_ylabel('time to failure', color='b')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)

plot_acc_ttf_data(train_ad_sample_df[::100], train_ttf_sample_df[::100])
print(len(train_ad_sample_df[::100]))
#del train_ad_sample_df
#del train_ttf_sample_df


# In[7]:


test_seg_00030f = pd.read_csv('C:\\Users\\talal\\Desktop\\WUT\\S3\\Machine Learning\\LANL-Earthquake-Prediction\\test\\seg_00030f.csv', dtype={'acoustic_data': np.int16})


# In[8]:


test_sample = test_seg_00030f['acoustic_data'].values[::]


# In[9]:


#function for plotting based on both features
def plot_acc_ttf_data(train_ad_sample_df,  title="Acoustic data"):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title(title)
    plt.plot(train_ad_sample_df, color='r')
    ax1.set_ylabel('acoustic data', color='r')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))
    plt.grid(True)


plot_acc_ttf_data(test_sample)
#del train_ad_sample_df
#del train_ttf_sample_df


# In[10]:


def plot_ad_his_del_range(train_ad_sample_df, min_range, max_range,n_bins, title="Histogram with deleted middle range"):
    train_ad_sample_df_filtered = [x for x in train_ad_sample_df if x <=min_range or max_range <= x ]

    bin_list_train_ad_sample_df = np.linspace(min(train_ad_sample_df_filtered),max(train_ad_sample_df_filtered),n_bins)
    plt.title(title)
    plt.hist(train_ad_sample_df_filtered, bins=bin_list_train_ad_sample_df)
plot_ad_his_del_range(train_ad_sample_df, -200, 200, 30)


# In[11]:


def plot_amp_diff(train_ad_sample_df_local, train_ttf_sample_df_local, skip_num,threshold_max, title="Scatter plot with amplitud (aoustic) vs difference (displacement) with ttf coloring indicator"):
    x = train_ad_sample_df_local[::skip_num]
    y = x.copy()
    for i in range(len(y)-1,0,-1):
        y[i] = y[i]-y[i-1]
    sample_train_ttf = train_ttf_sample_df_local.copy()[::skip_num]
    sample_train_ttf[(sample_train_ttf>threshold_max)]=threshold_max
    #print(str(len(x))+" "+str(len(sample_train_ttf)))
    min_sm = min(sample_train_ttf)
    max_sm = max(sample_train_ttf)
    rgb = (sample_train_ttf - min_sm)/ (max_sm - min_sm)

    #print(min(rgb)+max(rgb))
    #rgb = np.random.random((10, 3))
    #print(rgb)
    fig, ax = plt.subplots()
    #ax.scatter(x, y,c='0.1',)
    ax.scatter(x,y, c=rgb , marker = '.', cmap=cm.Greys)
    ax.set_facecolor('xkcd:salmon')
    ax.set_facecolor((1.0, 0.47, 0.42))
    plt.title(title)
    plt.show()
    del sample_train_ttf
    del y
plot_amp_diff(train_ad_sample_df,train_ttf_sample_df, 5,8)


# In[12]:


#Get the locations where ttf = 0
#train_ttf_sample_df_list - time to fail data
def get_list_ttf_spike(train_ttf_sample_df_list):
    tmp = train_ttf_sample_df_list<0.1
    t_spike=[iv for iv, ndx in enumerate(tmp) if ndx]
    #print(t_spike)
    ndx = 1
    time_spike=[]
    for ind in range(1,len(t_spike)):
        if(t_spike[ind-1]+1!=t_spike[ind]):
            time_spike.append(t_spike[ind])
    time_spike.append(t_spike[-1])
    return time_spike
print(get_list_ttf_spike(train_ttf_sample_df))


# In[13]:


#Location and intencity of the spike with respect to the time to fail location
#spike_ttf - the time to fail = 0 location (from get_list_ttf_spike)
#range_back - stepping value to go back to find the spike from the ttf point
#spike_value - the threshold value that will stop the search if we hit this value
#train_ad_sample_df_local - the acustic data
def get_spike_to_ttf_time(spike_ttf,range_back,spike_value, train_ad_sample_df_local):    
    spike_diff = 0
    back_step = range_back    
    while(spike_diff<spike_value):
        x = train_ad_sample_df_local[spike_ttf-back_step:spike_ttf]
        y = x.copy()
        for i in range(len(y)-1,0,-1):
            y[i] = y[i]-y[i-1]    
        spike_diff = max(y)-min(y)
        if(spike_diff<spike_value):
            del y
        back_step = back_step+range_back
    
    #plot_acc_ttf_data(x)
    return len(y)-int(((np.argmax(y)+np.argmin(y))/2)), spike_diff


#get_spike_to_ttf_time(get_list_ttf_spike(train_ttf_sample_df)[-1],500,1000, train_ad_sample_df)

for i in get_list_ttf_spike(train_ttf_sample_df):
    print(get_spike_to_ttf_time(i,500,1000, train_ad_sample_df))


# In[14]:


#Plotting Spike intensity vs delay to ttf
#spike_list - list of tupils (delay, spike intencity)
#train_ad_sample_df_local - acustic data set
def plot_spike_delay(spike_list,train_ad_sample_df_local, title="Spike intencity vs time to fail delays"):
    spikes = []
    delays = []
    for i in spike_list:
        r = get_spike_to_ttf_time(i,500,1000, train_ad_sample_df_local)
        delays.append(r[0])
        spikes.append(r[1])        
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title(title)    
    ax.set_xlabel('delay', color='b')
    ax.set_ylabel('spike', color='r')
    ax.scatter(delays,spikes, marker = 's')
    plt.grid(True)
plot_spike_delay(get_list_ttf_spike(train_ttf_sample_df),train_ad_sample_df)


# In[15]:


def plot_ad_his_only_mid_range(train_ad_sample_df, min_range, max_range,n_bins, title="Histogram with middle range only"):
    train_ad_sample_df_filtered = [x for x in train_ad_sample_df if x > min_range and max_range > x ]

    bin_list_train_ad_sample_df = np.linspace(min(train_ad_sample_df_filtered),max(train_ad_sample_df_filtered),n_bins)
    plt.title(title)
    plt.hist(train_ad_sample_df_filtered, bins=bin_list_train_ad_sample_df)
plot_ad_his_only_mid_range(train_ad_sample_df, -40, 40, 150)


# In[16]:


train.describe()


# In[205]:


#lets create a function to generate some statistical features based on the training data
def gen_features(X):
    strain = []
    strain.append(X.mean())
    strain.append(X.std())
    strain.append(X.min())
    strain.append(X.max())
    #strain.append(X.kurtosis())
    #strain.append(X.skew())
    strain.append(np.quantile(X,0.01))
    strain.append(np.quantile(X,0.05))
    strain.append(np.quantile(X,0.95))
    strain.append(np.quantile(X,0.99))
    strain.append(np.abs(X).max())
    strain.append(np.abs(X).mean())
    strain.append(np.abs(X).std())
    return pd.Series(strain)


# In[18]:


scaler_acoustic = MinMaxScaler(feature_range=(0, 1))
scaler_ttf = MinMaxScaler(feature_range=(0, 1))
train_float_ad = train_ad_sample_df.astype('float32').reshape(-1,1)
train_float_ttf = train_ttf_sample_df.astype('float32').reshape(-1,1)
train_normalized_ad = scaler_acoustic.fit_transform(train_float_ad)
train_normalized_ttf = scaler_ttf.fit_transform(train_float_ttf)


# In[19]:


print(train_normalized_ad.shape)
print(train_normalized_ttf.shape)


# In[28]:


# split into train and test sets
train_size = int(len(train_normalized_ad) * 0.67)
test_size = len(train_normalized_ad) - train_size
train_set_ad, test_set_ad = train_normalized_ad[0:train_size,:], train_normalized_ad[train_size:len(train_normalized_ad),:]
train_set_ttf, test_set_ttf = train_normalized_ttf[0:train_size,:], train_normalized_ttf[train_size:len(train_normalized_ttf),:]
print("acoustic - train: "+str(len(train_set_ad)),"test: " + str(len(test_set_ad)))
print("time to fail - train: "+str(len(train_set_ttf)),"test: " + str(len(test_set_ttf)))


# In[208]:


# convert an array of values into a dataset matrix
def create_dataset(dataset_ad,dataset_ttf, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset_ad)-look_back-1):
		a = dataset_ad[i:(i+look_back), 0]
		dataX.append(gen_features(a))#dataX.append(a)#
		b = dataset_ttf[i+look_back, 0]
		dataY.append(b)
	return np.array(dataX), np.array(dataY)


# In[209]:


# reshape into X=t and Y=t+1
look_back = 100
trainX, trainY = create_dataset(train_set_ad,train_set_ttf, look_back)
testX, testY = create_dataset(test_set_ad,test_set_ttf, look_back)


# In[212]:


#save to file
#with open('trainX_look_back_100_with_features', 'wb') as f:
    #pickle.dump(trainX, f)
#with open('trainY_look_back_100_with_features', 'wb') as f:
    #pickle.dump(trainY, f)
#with open('testX_look_back_100_with_features', 'wb') as f:
    #pickle.dump(testX, f)
#with open('testY_look_back_100_with_features', 'wb') as f:
    #pickle.dump(testY, f)
#read from file
with open('trainX_look_back_100_with_features', 'rb') as f:
    trainX = pickle.load(f)
with open('trainY_look_back_100_with_features', 'rb') as f:
    trainY = pickle.load(f)
with open('testX_look_back_100_with_features', 'rb') as f:
    testX = pickle.load(f)
with open('testY_look_back_100_with_features', 'rb') as f:
    testY = pickle.load(f)


# In[213]:


# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[214]:


print(train_set_ad[0])
print(len(trainY[::40]))


# In[216]:


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(10, input_shape=(1, trainX.shape[-1])))#input_shape=(1, look_back)
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX[::40], trainY[::40], epochs=5000, batch_size=32, verbose=2)


# In[136]:


#Creat and fit nn with relu activation and adam opt
def create_model(input_dim=11):
    model = Sequential()
    model.add(Dense(256, activation="relu",input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="linear"))
 
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # Compile model
    model.compile(
        loss='mae',
        optimizer=opt,
    )
    return model
model = create_model(trainX.shape[-1])
model.fit(trainX[::100], trainY[::100], epochs=500, batch_size=32, verbose=2)


# In[144]:


# CatBoost
#model = CatBoostRegressor(iterations=10000, loss_function='MAE', boosting_type='Ordered')
#model.fit(trainX[::100], trainY[::100], silent=True)


# In[158]:


#trainPredict, testPredict = [], []
#for i in range(len(trainX)):
    #a =  model.predict([trainX[i]])
    #trainPredict.append(a[0])
#for i in range(len(testX)):
    #a = model.predict([testX[i]])
    #testPredict.append(a[0])


# In[217]:


#import pickle
#save to file
with open('NN_LSTM_10_lookback_100_with_features_epochs_5000_batch_32_size_105380', 'wb') as f:
    pickle.dump(model, f)


# In[170]:


#read from file
#with open('NN_LSTM_4_lookback_10_epochs_100_size_70255', 'rb') as f:
    #copy_model = pickle.load(f)


# In[218]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler_ttf.inverse_transform(trainPredict)
trainY = scaler_ttf.inverse_transform(trainY.reshape(-1,1))
testPredict = scaler_ttf.inverse_transform(testPredict)
testY = scaler_ttf.inverse_transform(testY.reshape(-1,1))
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))


# In[219]:


# shift train predictions for plotting
trainPredictPlot = np.empty_like(train_normalized_ttf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(train_normalized_ttf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(train_normalized_ttf)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler_ttf.inverse_transform(train_normalized_ttf))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[172]:


print(np.array(trainPredict).reshape(-1,1).shape)


# In[20]:


train_chucnk = pd.read_csv('C:\\Users\\talal\\Desktop\\WUT\\S3\\Machine Learning\\LANL-Earthquake-Prediction\\train.csv', iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})


# In[ ]:


X_train = pd.DataFrame()
y_train = pd.Series()
for df in train_chucnk:
    ch = gen_features(df['acoustic_data'])
    X_train = X_train.append(ch, ignore_index=True)
    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))


# In[ ]:


X_train.describe()


# In[ ]:


#Model #1 - Catboost

train_pool = Pool(X_train, y_train)
m = CatBoostRegressor(iterations=10000, loss_function='MAE', boosting_type='Ordered')
m.fit(X_train, y_train, silent=True)
m.best_score_


# In[ ]:


pred_sample = m.predict([test_sample])


# In[ ]:


pred_sample


# In[ ]:




