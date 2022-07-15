import numpy as np
import tensorflow as tf

txt = open(r'C:\Users\ParkJeongSu\Documents\vsc\DL\piano\piano.txt','r').read()

# print(txt)

# 전처리
bow = list(set(txt))
bow.sort()
print(bow,len(bow))

txt_to_num = {}
num_to_txt = {}
preprocessTxt = []

for i,j in enumerate(bow):
    txt_to_num[j] = i
    num_to_txt[i] = j


for i in txt:
    preprocessTxt.append(txt_to_num[i])

# print(preprocessTxt)

train_x = []
train_y = []

for i in range(0,len(preprocessTxt)-25):
    train_x.append(preprocessTxt[i:i+25])
    train_y.append(preprocessTxt[i+25])

train_x = np.array(train_x)
train_y = np.array(train_y)

print(train_x)

train_x = tf.one_hot(train_x,len(bow))
train_y = tf.one_hot(train_y,len(bow))