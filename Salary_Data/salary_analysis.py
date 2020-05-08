#!/usr/bin/env python
# coding: utf-8

# # Análise de Regressão Linear Simples
# 
# ## Salário vs. Anos de Experiência
# 
# #### Exercício de Fixação dos conceitos e aplicação da regressão linear simples 

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


salary = pd.DataFrame(pd.read_csv('Salary_Data.csv'))


# In[6]:


salary.head()


# ## Inspeção nos dados

# In[7]:


salary.shape


# In[8]:


salary.info()


# In[9]:


salary.describe()


# ### Limpeza de Dados

# In[10]:


# Verificação de Dados Faltantes ou nulos
salary.isnull().sum()*100/salary.shape[0]


# ### Análise de Outliers

# In[12]:


sns.boxplot(y = salary['YearsExperience'])


# In[13]:


sns.boxplot( y = salary['Salary'])


#  Não existem pontos considerados como Outliers no Dataset, tanto em YearsExperience quanto em Salary

# ## Análise Exploratória
# Verificando a disperção dos pontos

# In[19]:


sns.scatterplot(salary['YearsExperience'], salary['Salary'])


# Visualmente, podemos inferir uma correlação linear positiva entre as duas variáveis.

# In[21]:


sns.heatmap(salary.corr(), cmap="YlGnBu", annot = True)
plt.show()


# Coeficiente de correlação satisfatório para investigação de uma regressão linear simples.

# ## Construção do Modelo de Regressão Linear Simples.

# In[55]:


X = salary['YearsExperience']
Y = salary['Salary']


# In[56]:


from sklearn.model_selection import train_test_split


# In[57]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[59]:


import statsmodels.api as sm
X_opt = sm.add_constant(Xtrain)
regressor = sm.OLS(Ytrain, X_opt).fit()


# In[60]:


regressor.params


# In[61]:


print(regressor.summary())


# ### Parametros importantes para a análise
# 
# R²
# 
# F estatístico
# 
# 1 - O coeficiente de 'YearsExperience' é 9379.7105 com um valor p muito baixo
# 
# 2 - R² é 0.940
# 
# Significa que 94% da variância de Salary é explicada por YearsExperience
# 
# Com esse parâmetros e análises, podemos avaliar graficamente a regressão linear

# In[66]:


plt.scatter(Xtrain, Ytrain)
plt.plot(Xtrain, 26990 + 9379.7105 * Xtrain, color = 'red')
plt.show()


# In[75]:


plt.scatter(Xtrain, res)


# Tendo em vista que o gráfico acima apresenta pontos bem dispersos, temos que o erro se mantém estável à medida que cresce
# o valor de 'YearsExperience'.
# Agora precisamos fazer previsões da base de test (Xtest) utilizando o modelo de previsão construído na base de treino.

# In[77]:


# Adicionando a constante no modelo de regressão
Xtest_sm = sm.add_constant(Xtest)

# Fazendo previsões dos valores y correspondentes ao Xtest_sm
Ypred = regressor.predict(Xtest_sm)


# In[78]:


Ypred.head()


# In[79]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[80]:


np.sqrt(mean_squared_error(Ytest, Ypred))


# In[81]:


r_squared = r2_score(Ytest, Ypred)


# In[82]:


r_squared


# In[83]:


plt.scatter(Xtest, Ytest)
plt.plot(Xtest, 26990 + 9379.7105 * Xtest)
plt.show()


# In[ ]:




