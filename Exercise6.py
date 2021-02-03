#!/usr/bin/env python
# coding: utf-8

# In[107]:


#Loading MNIST

import pprint
import numpy as np
import tensorflow as tf
import gzip, pickle ;
import matplotlib.pyplot as plt
DIR = 'C:\\Users\\Saksham\\Downloads\\MNIST.pkl.gz'
with gzip.open(DIR) as f:
    data = pickle.load(f) ;
    traind = data["data_train"].reshape(-1,28,28)
    trainl = data["labels_train"]


# In[108]:


#a. create scatter-plot of the image-wise pixel mean for all samples. 

means = traind.mean(axis=(1,2))
x = np.arange(55000,)
plt.scatter(x, means)
plt.show()


# In[119]:


#b copy out all the samples whose mean over pixels is > 0.3 and display 3 of them

meanb = traind.mean(axis=(1,2))
mask = meanb > 0.3
cc = traind[mask]
print(mask.shape, cc.shape)
fig = plt.figure()
fig.add_subplot(1,3,1)
plt.imshow(cc[0])
fig.add_subplot(1,3,2)
plt.imshow(cc[1])
fig.add_subplot(1,3,3)
plt.imshow(cc[2])
plt.show()


# In[125]:


#c Compute the “average image” and display it!

avg = traind.mean(axis=0)
plt.imshow(avg)
plt.show()


# In[126]:


#d Compute the “average image” for samples of class 5 and display it!

class5 = trainl.argmax(axis=1)
mask = (class5 == 5)
avg = traind[mask].mean(axis=0)
plt.imshow(avg)
plt.show()


# In[9]:


#SOFTMAX USING NUMPY
# exp = 2.718281

def S(x):
    e = np.exp(x)
    return e/e.sum()

x1 = np.array([-1,-1,5, 1, 1, 2]).reshape(2,3,1)
x3 = np.array([-1,-1,5])
x2 = np.array([1,1,2])
        
print(S(x1[0,:,:]))
print(S(x3))
print(S(x2))


# In[19]:


#SOFTMAX USING TENSORFLOW

def S(x):
    e = tf.math.exp(x)
    return e/tf.reduce_sum(e)

x1 = tf.constant(x1, dtype =tf.float64)
x2 = tf.constant([1,1,2], dtype =tf.float64)

print(S(x1).numpy()[0])
print(S(x2).numpy())


# In[24]:


#CROSS-ENTROPY using numpy & tensorflow

def CE(y, t): #tensor
    return tf.reduce_sum(-tf.math.log(y)*t)

def CENP(y, t): #numpy
    return (-np.log(y)*t).sum()

t = np.array([0,0,1])
y1 = np.array([0.1,0.1,0.8])
y2 = np.array([0.3,0.3,0.4])
y3 = np.array([0.8,0.1,0.1])

print(CE(y1, t).numpy())
print(CE(y2, t).numpy())
print(CE(y3, t).numpy())

print('-------------')

print(CENP(y1, t))
print(CENP(y2, t))
print(CENP(y3, t))


