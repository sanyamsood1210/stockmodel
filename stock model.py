#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


hello=pd.read_csv(r'Downloads/GME_stock.csv')


# In[3]:


df=hello.reset_index()['close_price']


# In[4]:


df


# In[5]:


plt.plot(df)


# In[6]:


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df=sc.fit_transform(np.array(df).reshape(-1,1))


# In[7]:


training_size=int(len(df)*0.65)
train_size=len(df)-training_size
train_data,test_data=df[0:training_size,:],df[training_size:len(df),:1]


# In[8]:


training_size


# In[10]:


import numpy
def create_dataset(dataset,time_step=1):
    datax,datay=[], []
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        datax.append(a)
        datay.append(dataset[i+time_step,0])
    return numpy.array(datax),numpy.array(datay)


# In[11]:


time_step=100
x_train,y_train=create_dataset(train_data,time_step)
x_test,y_test=create_dataset(test_data,time_step)


# In[12]:


x_train=x_train.reshape(x_train.shape[0],x_train.shape[1] ,1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1] ,1)


# In[13]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[14]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[15]:


model.summary()


# In[16]:


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=64,verbose=1)


# In[17]:


train_predict=model.predict(x_train)


# In[18]:


test_p=model.predict(x_test)


# In[19]:


train_predict=sc.inverse_transform(train_predict)


# In[20]:


test_p=sc.inverse_transform(test_p)


# In[21]:


import math


# In[22]:


from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[23]:


math.sqrt(mean_squared_error(y_test,test_p))


# In[24]:


look_back=100
trainPredictPlot=numpy.empty_like(df)
trainPredictPlot[:, :]=np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :]=train_predict
testPredictPlot=numpy.empty_like(df)
testPredictPlot[:, :]=numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df)-1, :]=test_p
plt.plot(sc.inverse_transform(df))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[25]:


len(test_data)


# In[26]:


x_input=test_data[1571:].reshape(-1,1)


# In[27]:


x_input.shape


# In[28]:


temp_input=list(x_input)


# In[29]:


temp_input=temp_input[0].tolist()


# In[30]:


from numpy import array
list_output=[]
n_steps=100
i=0
while(i<30):
    if(len(temp_input)>100):
        x_input=np.array(team_input[1:])
        print(" {} day input{}".format(i,x_input))
        x_input=x_input.reshape(-1,1)
        x_input=x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print("{} day output".format(i,yhat))
        temp_input.extend( yhat[0].tolist())
        temp_input=temp_input[1:]
        list_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input=x_input.reshape((1, n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        list_output.extend(yhat.tolist())
        i=i+1
print(list_output)
        


# In[31]:


day=np.arange(1,101)
predict=np.arange(101,131)


# In[32]:


dataf=df.tolist()
dataf.extend(list_output)
plt.plot(dataf[4500:])


# In[33]:


plt.plot(day,sc.inverse_transform(df[4673:]))
plt.plot(predict,sc.inverse_transform(list_output))


# In[ ]:




