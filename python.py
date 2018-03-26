# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 21:49:32 2018

@author: www
"""

from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random


import string

digits = string.digits
opr = '+-*'

#数据生成器
def gen():
     s=''
     n = np.random.randint(0, 3)
     if n==0:
          s=(s+'('+random.choice(digits)+random.choice(opr)
          +random.choice(digits)+')'+random.choice(opr)+random.choice(digits))
     elif n==1:
          s=(s+random.choice(digits)+random.choice(opr)+'('
          +random.choice(digits)+random.choice(opr)+random.choice(digits)+')')
     elif n==2:
          s=(s+random.choice(digits)+random.choice(opr)
          +random.choice(digits)+random.choice(opr)+random.choice(digits))
     return s


characters = string.digits + opr + '()'
width, height, n_len, n_class = 180, 60, 7, len(characters)

def generator(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.float32)
    y = [np.zeros((batch_size, n_class), dtype=np.float32) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            s=gen()
            X[i] = generator.generate_image(s)
            for j, ch in enumerate(s):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y

X, y = next(generator(1))



#定义网络结构
from keras.models import *
from keras.layers import *

input_tensor = Input((height, width, 3))
x = input_tensor
for i in range(3):
    x = Conv2D(32*2**i, (3,3), activation='relu')(x)
    x = Conv2D(32*2**i, (3,3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(7)]
model = Model(input=input_tensor, output=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit_generator(generator(), samples_per_epoch=64, nb_epoch=1,
                    validation_data=generator(),validation_steps=32)


#读入测试集进行预测
from tqdm import tqdm
import pandas as pd
import cv2
path = r'F:\data\image_contest_level_1'
df = pd.read_csv(path+'\image_contest_level_1\labels.txt', sep=' ', header=None,names=['a','b'])

n_test = 100000
X_test = np.zeros((n_test, height, width, 3), dtype=np.float32)
y_test = np.zeros((n_test, n_len), dtype=np.float32)
label_length_test = np.zeros((n_test, 1), dtype=np.float32)

for i in tqdm(range(n_test)):
     img = cv2.imread(path+'\image_contest_level_1\%d.png'%i)
     X_test[i] = img
     random_str = df['a'][i]
     y_test[i,:len(random_str)] = [characters.find(x) for x in random_str]
     y_test[i,len(random_str):] = -1


#解码器
def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return y

X, y = next(generator(1))
y_pred = model.predict(X)
plt.title('real: %s\npred:%s'%(decode(y), decode(y_pred)))
plt.imshow(X[0], cmap='gray')
plt.axis('off')

from tqdm import tqdm
def evaluate(model, batch_num=20):
    batch_acc = 0
    generator = gen()
    for i in tqdm(range(batch_num)):
        X, y = generator.next()
        y_pred = model.predict(X)
        batch_acc += np.mean(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T))
    return batch_acc / batch_num

evaluate(model)

model.save('cnn.h5')    
          


                 

     

     





