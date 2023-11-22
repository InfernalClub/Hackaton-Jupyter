#!/usr/bin/env python
# coding: utf-8

# # Desafío 

# ## 1. Carga de libreria y Datos de estudio

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
# Displays the plots for us.
get_ipython().run_line_magic('matplotlib', 'inline')


### Multicolinealidad 


# In[2]:


df = pd.read_excel("Hack_concentraducto_v01.xlsx")
df


# In[3]:


df.shape


# ## 2.  Definición de rangos y filtración de datos(max, min e histrograma)

# ### % Solido Bombeo concentrado_EB
# 

# In[4]:


df['% Solido Bombeo concentrado_EB'].max()


# In[5]:


df['% Solido Bombeo concentrado_EB'].min()


# In[6]:


df[["% Solido Bombeo concentrado_EB"]].hist()


# ### Presión de Descarga_EB_1
# 

# In[7]:


df[["Presión de Descarga_EB_1"]].hist()


# ### Presión de Descarga_EB_2
# 

# In[8]:


df['Presión de Descarga_EB_2'].max()


# In[9]:


df['Presión de Descarga_EB_2'].min()


# In[10]:


df[["Presión de Descarga_EB_2"]].hist()


# ### Presion_Estación de Valvulas_EV1_1
# 

# In[11]:


df['Presion_Estación de Valvulas_EV1_1'].max()


# In[12]:



df['Presion_Estación de Valvulas_EV1_1'].min()


# In[13]:


df[["Presion_Estación de Valvulas_EV1_1"]].hist()


# ### Presion_Estación de Valvulas_EV1_2
# 

# In[14]:


df['Presion_Estación de Valvulas_EV1_2'].max()


# In[15]:


df['Presion_Estación de Valvulas_EV1_2'].min()


# In[16]:


df[["Presion_Estación de Valvulas_EV1_2"]].hist()# 


# ### Presión_SM-1
# 

# In[17]:


df['Presión_SM-1'].max()


# In[18]:


df['Presión_SM-1'].min()


# In[19]:


df[["Presión_SM-1"]].hist()


# ### Presión_SM-2
# 

# In[20]:


df['Presión_SM-2'].max()


# In[21]:


df['Presión_SM-2'].min()


# In[22]:


df[["Presión_SM-2"]].hist()


# ### Presión estación de valvulas 2_EV2_1
# 

# In[23]:


df['Presión estación de valvulas 2_EV2_1'].max()


# In[24]:


df['Presión estación de valvulas 2_EV2_1'].min()


# In[25]:


df[["Presión estación de valvulas 2_EV2_1"]].hist()


# ### Presión estación de valvulas 2_EV2_2
# 

# In[26]:


df['Presión estación de valvulas 2_EV2_2'].max()


# In[27]:


df['Presión estación de valvulas 2_EV2_2'].min()


# In[28]:


df[["Presión estación de valvulas 2_EV2_2"]].hist()


# ### Porcentaje de Solido Alimentación Espesador
# 

# In[29]:


df['Porcentaje de Solido Alimentación Espesador'].max()


# In[30]:


df['Porcentaje de Solido Alimentación Espesador'].min()


# In[31]:


df[["Porcentaje de Solido Alimentación Espesador"]].hist()


# ### Presión_EDT_1
# 

# In[32]:


df['Presión_EDT_1'].max()


# In[33]:


df['Presión_EDT_1'].min()


# In[34]:


df[["Presión_EDT_1"]].hist()# se observa que los valores de estado_bomba_BHC2 son 0 po ende, no se incluira en el modelo 


# ### Presión_EDT_2
# 

# In[35]:


df['Presión_EDT_2'].max()


# In[36]:


df['Presión_EDT_2'].min()


# In[37]:


df[["Presión_EDT_2"]].hist() # de 80 a 100 


# ### Presión_EDT_3
# 
# 

# In[38]:


df['Presión_EDT_3'].max()


# In[39]:


df['Presión_EDT_3'].min()


# In[40]:


df[["Presión_EDT_3"]].hist()


# ### Presión_SM-3
# 

# In[41]:


df['Presión_SM-3'].max()


# In[42]:


df['Presión_SM-3'].min()


# In[43]:


df[["Presión_SM-3"]].hist() 


# ### Presión_SM-4

# In[44]:


df['Presión_SM-4'].max()


# In[45]:


df['Presión_SM-4'].min()


# In[46]:


df[["Presión_SM-4"]].hist() 


# ## 3.  Grafica de dispersión de los datos

# In[47]:


x_column = ['% Solido Bombeo concentrado_EB','Presión de Descarga_EB_1','Presión de Descarga_EB_2',"Presion_Estación de Valvulas_EV1_1"]
y_column = ['Presión_SM-1']
sns.pairplot(df,x_vars=x_column,y_vars=y_column)
plt.show()


# In[48]:


x_column = ['Presion_Estación de Valvulas_EV1_2','Presión estación de valvulas 2_EV2_1','Presión estación de valvulas 2_EV2_2',"Porcentaje de Solido Alimentación Espesador"]##2
y_column = ['Presión_SM-1']
sns.pairplot(df,x_vars=x_column,y_vars=y_column)
plt.show()


# In[49]:


x_column = ['Presión_EDT_1','Presión_EDT_2','Presión_EDT_3']##3
y_column = ['Presión_SM-1']
sns.pairplot(df,x_vars=x_column,y_vars=y_column)
plt.show()


# ## 5. Modelo predictivo SM-1
# 

# ### 5.1 Deficion de los datos en una muestra de entrenamiento y  una de prueba 

# In[50]:


columnas_modelo =['% Solido Bombeo concentrado_EB','Presión de Descarga_EB_1','Presión de Descarga_EB_2','Presion_Estación de Valvulas_EV1_1','Presion_Estación de Valvulas_EV1_2','Presión_SM-1','Presión_SM-2','Presión estación de valvulas 2_EV2_1','Presión estación de valvulas 2_EV2_2','Porcentaje de Solido Alimentación Espesador','Presión_EDT_1','Presión_EDT_2','Presión_EDT_3','Presión_SM-3','Presión_SM-4']
columnas_predictoras = ['% Solido Bombeo concentrado_EB','Presión de Descarga_EB_1','Presión de Descarga_EB_2','Presion_Estación de Valvulas_EV1_1','Presion_Estación de Valvulas_EV1_2','Presión_SM-1','Presión_SM-2','Presión estación de valvulas 2_EV2_1','Presión estación de valvulas 2_EV2_2','Porcentaje de Solido Alimentación Espesador','Presión_EDT_1','Presión_EDT_2','Presión_EDT_3','Presión_SM-4']
columna_dependiente = ['Presión_SM-3']
df_model = df[columnas_modelo].copy()


# In[ ]:





# In[51]:


#set random_state to get the same split every time
df_test_size = 0.2
traindf, testdf = train_test_split(df_model, test_size=df_test_size, random_state=42) #para reproducir resultad


# In[52]:


# testing set es cercano al 20% de los datos; conjunto de entrenamiento alrededor del 80%
print("Shape del dataset completo:",df_model.shape,)
print("Shape del dataset de entrenamiento: ",traindf.shape)
print("Shape del dataset de prueba:",testdf.shape)


# In[53]:


traindf.head()


# ### 5.2 Modelo de Regresion Multivaraible OILS 
# Creamos el modelo de regresión utilizando el modelo de los mínimos cuadrados ordinarios del cual analizaremos los coeficientes de las variables independientes y el R2

# In[54]:


# El vector y corresponde a la variable que queremos predecir: Presion BHC2
y_train_lr = traindf[columna_dependiente].copy()
x_train_lr = traindf[columnas_predictoras].copy()

y_test_lr = testdf[columna_dependiente].copy()
x_test_lr = testdf[columnas_predictoras].copy()


# In[55]:


#crear modelo lineal
regression =LinearRegression()

#fit
est = sm.OLS(y_train_lr, x_train_lr)
est2 = est.fit()
print(est2.summary())


# In[56]:


print('MAE train:',mean_absolute_error(y_train_lr, est2.predict(x_train_lr)))


# In[57]:


prediction_train = est2.predict(x_train_lr)
ax = sns.regplot(y_train_lr,prediction_train.to_frame('Predicción Train'))
r2_score(y_train_lr, prediction_train)


# In[58]:


#Predecimos
y_test_pred = est2.predict(x_test_lr)


# In[59]:


print('MAE test:',mean_absolute_error(y_test_lr, est2.predict(x_test_lr)))


# In[60]:


MAE_Train = mean_absolute_error(y_train_lr, est2.predict(x_train_lr))
MAE_Test = mean_absolute_error(y_test_lr, est2.predict(x_test_lr))
objective_tag = "Presión_SM-1"
Y_train_data = y_train_lr
y_pred_train = prediction_train
Y_test_data = y_test_lr
y_pred_test = y_test_pred 

xx = np.linspace(2,8)
yy = xx

Texto1 = 'Train_MAE:'+str(round(MAE_Train,3))
Texto2 = 'Test_MAE:'+str(round(MAE_Test,3))
Texto3 = objective_tag+' Real vs Predicha \n Entrenamiento'
Texto4 = objective_tag+' Real vs Predicha \n Test'
Texto5 = objective_tag+' Real'
Texto6 = objective_tag+' Predicha'
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,2,1)
ax.scatter(Y_train_data, y_pred_train, marker='o', color='blue')
ax.plot(xx,yy,color='black', alpha=0.6, linewidth=3.0)
ax.annotate(Texto1, xy=(0.02, 0.95), xycoords='axes fraction', color='black', fontsize=13,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.25'))
ax.grid(True)
ax.set_xlabel(Texto5,fontsize=16)
ax.set_ylabel(Texto6,fontsize=16)
ax.set_title(Texto3, fontsize=16)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

ax = fig.add_subplot(1,2,2)
ax.scatter(Y_test_data, y_pred_test, marker='o', color='blue')
ax.plot(xx,yy,color='black', alpha=0.6, linewidth=3.0)
ax.annotate(Texto2, xy=(0.02, 0.95), xycoords='axes fraction', color='black', fontsize=13,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.25'))
ax.grid(True)
ax.set_xlabel(Texto5,fontsize=16)
ax.set_ylabel(Texto6,fontsize=16)
ax.set_title(Texto4, fontsize=16)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

plt.tight_layout()
plt.show()


# In[61]:


#ax = sns.regplot(y_test_lr,y_test_pred.to_frame('Predicción Test'))
print("El R2 Score de la predicción en el conjunto de test fue:" +str(round(r2_score(y_test_lr, y_test_pred),2)))


# ### 5.3 Predictores estadisticamente significativos segun el modelo OILS

# Por criterio de valor P se descarta % Solido Bombeo concentrado_EB

# ### 5.4 Modelo de Regresion Multivaraible OILS con variables predictoras
# Creamos el modelo de regresión utilizando el modelo de los mínimos cuadrados ordinarios del cual analizaremos los coeficientes de las variables independientes y el R2

# In[62]:


columnas_modelo =['Presión de Descarga_EB_1','Presión de Descarga_EB_2','Presion_Estación de Valvulas_EV1_1','Presion_Estación de Valvulas_EV1_2','Presión_SM-1','Presión_SM-2','Presión estación de valvulas 2_EV2_1','Presión estación de valvulas 2_EV2_2','Porcentaje de Solido Alimentación Espesador','Presión_EDT_1','Presión_EDT_2','Presión_EDT_3','Presión_SM-3','Presión_SM-4']
columnas_predictoras = ['Presión de Descarga_EB_1','Presión de Descarga_EB_2','Presion_Estación de Valvulas_EV1_1','Presion_Estación de Valvulas_EV1_2','Presión_SM-1','Presión_SM-2','Presión estación de valvulas 2_EV2_1','Presión estación de valvulas 2_EV2_2','Porcentaje de Solido Alimentación Espesador','Presión_EDT_1','Presión_EDT_2','Presión_EDT_3','Presión_SM-4']
columna_dependiente = ['Presión_SM-3']
df_model = df[columnas_modelo].copy()


# In[63]:


#set random_state to get the same split every time
df_test_size = 0.2
traindf, testdf = train_test_split(df_model, test_size=df_test_size, random_state=42) #para reproducir resultad


# In[64]:


# testing set es cercano al 20% de los datos; conjunto de entrenamiento alrededor del 80%
print("Shape del dataset completo:",df_model.shape,)
print("Shape del dataset de entrenamiento: ",traindf.shape)
print("Shape del dataset de prueba:",testdf.shape)


# In[65]:


traindf.head()


# In[66]:


# El vector y corresponde a la variable que queremos predecir: Presion BHC1
y_train_lr = traindf[columna_dependiente].copy()
x_train_lr = traindf[columnas_predictoras].copy()

y_test_lr = testdf[columna_dependiente].copy()
x_test_lr = testdf[columnas_predictoras].copy()


# In[67]:


#### crear modelo lineal
regression =LinearRegression()

#fit
est = sm.OLS(y_train_lr, x_train_lr)
est2 = est.fit()
print(est2.summary())


# In[ ]:





# ### 5.5  Errores del modelamiento en el conjunto de training 

# Se determina el Error Absoluto Medio (MAE) para determinar la precisión del modelo.

# In[68]:


print('MAE train:',mean_absolute_error(y_train_lr, est2.predict(x_train_lr)))


# ### 5.6  Ajuste en el conjunto de entrenamiento

# In[69]:


prediction_train = est2.predict(x_train_lr)
ax = sns.regplot(y_train_lr,prediction_train.to_frame('Predicción Train'))
r2_score(y_train_lr, prediction_train)


# ### 5.7 Principales Conclusiones del Modelo:

# ### 5.8 Ejecución de  la predicción de los datos en el conjunto de test

# In[70]:


#Predecimos
y_test_pred = est2.predict(x_test_lr)


# In[71]:


print('MAE test:',mean_absolute_error(y_test_lr, est2.predict(x_test_lr)))


# In[72]:


MAE_Train = mean_absolute_error(y_train_lr, est2.predict(x_train_lr))
MAE_Test = mean_absolute_error(y_test_lr, est2.predict(x_test_lr))
objective_tag = "Presión_SM-1"
Y_train_data = y_train_lr
y_pred_train = prediction_train
Y_test_data = y_test_lr
y_pred_test = y_test_pred 

xx = np.linspace(2,8)
yy = xx

Texto1 = 'Train_MAE:'+str(round(MAE_Train,3))
Texto2 = 'Test_MAE:'+str(round(MAE_Test,3))
Texto3 = objective_tag+' Real vs Predicha \n Entrenamiento'
Texto4 = objective_tag+' Real vs Predicha \n Test'
Texto5 = objective_tag+' Real'
Texto6 = objective_tag+' Predicha'
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,2,1)
ax.scatter(Y_train_data, y_pred_train, marker='o', color='blue')
ax.plot(xx,yy,color='black', alpha=0.6, linewidth=3.0)
ax.annotate(Texto1, xy=(0.02, 0.95), xycoords='axes fraction', color='black', fontsize=13,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.25'))
ax.grid(True)
ax.set_xlabel(Texto5,fontsize=16)
ax.set_ylabel(Texto6,fontsize=16)
ax.set_title(Texto3, fontsize=16)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

ax = fig.add_subplot(1,2,2)
ax.scatter(Y_test_data, y_pred_test, marker='o', color='blue')
ax.plot(xx,yy,color='black', alpha=0.6, linewidth=3.0)
ax.annotate(Texto2, xy=(0.02, 0.95), xycoords='axes fraction', color='black', fontsize=13,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.25'))
ax.grid(True)
ax.set_xlabel(Texto5,fontsize=16)
ax.set_ylabel(Texto6,fontsize=16)
ax.set_title(Texto4, fontsize=16)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

plt.tight_layout()
plt.show()


# In[73]:


#ax = sns.regplot(y_test_lr,y_test_pred.to_frame('Predicción Test'))
print("El R2 Score de la predicción en el conjunto de test fue:" +str(round(r2_score(y_test_lr, y_test_pred),2)))


# # Prueba Data Set Hackaton

# In[74]:


df = pd.read_excel("Data_test_hakcathon_CEN.xlsx",3)
df


# In[75]:


columnas_modelo =['Presión de Descarga_EB_1','Presión de Descarga_EB_2','Presion_Estación de Valvulas_EV1_1','Presion_Estación de Valvulas_EV1_2','Presión_SM-1','Presión_SM-2','Presión estación de valvulas 2_EV2_1','Presión estación de valvulas 2_EV2_2','Porcentaje de Solido Alimentación Espesador','Presión_EDT_1','Presión_EDT_2','Presión_EDT_3','Presión_SM-3','Presión_SM-4']
columnas_predictoras = ['Presión de Descarga_EB_1','Presión de Descarga_EB_2','Presion_Estación de Valvulas_EV1_1','Presion_Estación de Valvulas_EV1_2','Presión_SM-1','Presión_SM-2','Presión estación de valvulas 2_EV2_1','Presión estación de valvulas 2_EV2_2','Porcentaje de Solido Alimentación Espesador','Presión_EDT_1','Presión_EDT_2','Presión_EDT_3','Presión_SM-4']
columna_dependiente = ['Presión_SM-3']
df_model = df[columnas_modelo].copy()


# In[76]:


print("Shape del dataset completo:",df_model.shape,)
print("Shape del dataset de entrenamiento: ",traindf.shape)
print("Shape del dataset de prueba:",testdf.shape)


# In[77]:


testdf=df
testdf


# In[78]:


y_train_lr = traindf[columna_dependiente].copy()
x_train_lr = traindf[columnas_predictoras].copy()

y_test_lr = testdf[columna_dependiente].copy()
x_test_lr = testdf[columnas_predictoras].copy()


# In[79]:


print("Shape del dataset completo:",df_model.shape,)
print("Shape del dataset de entrenamiento: ",traindf.shape)
print("Shape del dataset de prueba:",testdf.shape)


# In[80]:


#Predecimos
y_test_pred = est2.predict(x_test_lr)
y_test_pred


# In[81]:


#df.to_excel("Prueba Resultado.xlsx")
y_test_pred.to_excel("Resultado test SM3.xlsx")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




