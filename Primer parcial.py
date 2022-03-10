#!/usr/bin/env python
# coding: utf-8

# # Parcial corte 1
# 
# ## Hector Diaz 

# In[1]:


import io
import sys
PATH = '/home/Elian,john y hector/Data'
DIR_DATA = '../Data/'
sys.path.append(PATH) if PATH not in list(sys.path) else None
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sys import getsizeof
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
filename = DIR_DATA + 'precios.csv'


# In[2]:


df=pd.read_csv(filename,sep=';',encoding = 'unicode_escape')
dfr=df[['product_name','m','price','pdp_url','brand_name','product_category','retailer','description','rating','review_count','style_attributes','total_sizes','available_size','color']]
dfr.head(8)


# In[3]:


#Informacion del dataset.
dfr.dtypes


# In[9]:


#Eliminaremos todas esas filas vacias. 
dfr = dfr.dropna(how='all')


# In[10]:


#Datos vacios.
dfr.isnull().any()


# In[ ]:


#Normalización de los registros 
dfr['product_name'] = dfr['product_name'].fillna('None')
dfr['m'] = dfr['m'].fillna(0.0)
dfr['price'] = dfr['price'].fillna(0.0)
dfr['pdp_url'] = dfr['pdp_url'].fillna('None')
dfr['brand_name'] = dfr['brand_name'].fillna('None')
dfr['product_category'] = dfr['product_category'].fillna('None')
dfr['retailer'] = dfr['retailer'].fillna('None')
dfr['description'] = dfr['description'].fillna('None')
dfr['rating'] = dfr['rating'].fillna(0.0)
dfr['review_count'] = dfr['review_count'].fillna(0.0)
dfr['style_attributes'] = dfr['style_attributes'].fillna('None')
dfr['available_size'] = dfr['available_size'].fillna('None')
dfr['color'] = dfr['color'].fillna('None')


# In[12]:


dfr['price'].astype(str)
price=[]
for x in dfr['price'].values:
    if type(x)==str:
        x=x.replace(u"Â\xa0","")
        x=x.replace('"',"")
        price.append(float(x))
    else:
        price.append(x)


# In[14]:


dfr['m'].astype(str)
m=[]
s=0
for x in dfr['m'].values:
    if type(x)==str:
        x=x.replace(u"Â\xa0","")
        x=x.replace('"',"")
        m.append(float(x))
    else:
        m.append(x)

        
#Comprobamos si los cambios funcionaron.
print(len(price),len(m))


# In[15]:


#Cambio de manera manual
dfr['m']=m
dfr['price']=price

dfr.head(8)


# # Analisis de marca.

# In[17]:


#distribucion del precio
f,ax = plt.subplots(figsize=(23,6))
ax = sns.distplot(dfr['price'],rug=True)
plt.show()


# In[18]:


dfr['brand_name'].value_counts().plot(kind="bar",figsize=(10,10))


# In[22]:


#Rating por marcas
f,ax = plt.subplots(figsize=(35,6))
ax = sns.boxplot(x='brand_name',y='rating',data=dfr)
plt.show()


# In[23]:


#Distribucion del precio para cada marca
f,ax = plt.subplots(figsize=(75,30))
ax = sns.boxplot(x='brand_name',y='price',data=dfr)
plt.show()


# In[35]:


#Colores usados por cada marca
dfr['color'].value_counts()


# In[36]:


marca=list(dfr['brand_name'])
color=list(dfr['color'])
vs=[]
vsp=[]
hp=[]
w=[]
ae=[]
for x in range(len(color)):
    if marca[x]=="Victoria's Secret":
        vs.append(color[x])
    elif marca[x]=="Victoria's Secret Pink":
        vsp.append(color[x])
    elif marca[x]=="HankyPanky":
        hp.append(color[x])
    elif marca[x]=="Wacoal":
        w.append(color[x])
    elif marca[x]=="AERIE":
        ae.append(color[x])


# In[37]:


import operator
def conteo(lista):
    repetido={}
    orden=[]
    for x in lista:
        if x in repetido:
            repetido[x]+=1
        else:
            repetido[x]=1
    dict_sort = sorted(repetido.items(), key=operator.itemgetter(1), reverse=True)
    for value in enumerate(dict_sort):
        orden.append(str(value[1][0])+":"+ str(repetido[value[1][0]]))
    return orden


# In[38]:


#Que colores y usados cuantas veces por Victoria Secrect
conteo(vs)


# In[39]:


#Que colores y usados cuantas veces por Victoria Secrect Pink
conteo(vsp)


# In[40]:


#Que colores y usados cuantas veces por HankyPanky.
conteo(hp)


# In[ ]:




