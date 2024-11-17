import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.stats import norm

"""# **Punto 1: Generador de 100 muestras de tamaño 10, 20, 50 con distribucion exponencial con lambda=10**
"""
#PUNTO 1
df2 = pd.DataFrame({"n=10":[], "n=20":[], "n=50":[]})
lambda_ = 10
min_ = 0
max_ = expon.ppf(0.999, scale=1/lambda_)
gran = 100
x = np.linspace(min_, max_, gran)

for k in range (0,100):
  num_clases4 = int(round(1 + 3.3 * np.log(10), 0))
  y1 = expon.rvs(scale=1/lambda_, size=10)
  if (k<3):
    df2_1 = pd.DataFrame({"n=10":[]})
    for j in range (0, y1.size):
      nuevo_registro = {"n=10":y1[j]}
      df2_1=df2_1.append(nuevo_registro, ignore_index=True)

    print("curtosis grafica: ",k+1, "funcion exponencial ", df2_1.kurt())

  num_clases5 = int(round(1 + 3.3 * np.log(20), 0))
  y2 = expon.rvs(scale=1/lambda_, size=20)
  if (k<3):
    df2_1 = pd.DataFrame({"n=20":[]})
    for j in range (0, y2.size):
      nuevo_registro = {"n=20":y2[j]}
      df2_1=df2_1.append(nuevo_registro, ignore_index=True)

    print("curtosis grafica: ",k+1, "funcion exponencial ", df2_1.kurt())

  num_clases6 = int(round(1 + 3.3 * np.log(50), 0))
  y3 = expon.rvs(scale=1/lambda_, size=50)
  if (k<3):
    df2_1 = pd.DataFrame({"n=50":[]})
    for j in range (0, y3.size):
      nuevo_registro = {"n=50":y3[j]}
      df2_1=df2_1.append(nuevo_registro, ignore_index=True)
    print("curtosis grafica: ",k+1, "funcion exponencial ", df2_1.kurt())

  nuevo_registro = {"n=10":y1,"n=20":y2,"n=50":y3}

  df2=df2.append(nuevo_registro, ignore_index=True)

#EXPORTAR DATOS CON LAS 8000 FILAS
df2.to_csv("Datos_expon.csv")

"""**Histograma de frecuencias relativas para un n=10**"""

from scipy.stats import skew

from numpy import var

lambda_ = 10
min_ = 0
max_ = expon.ppf(0.999, scale=1/lambda_)
gran = 100
x = np.linspace(min_, max_, gran)
for k in range (0,3):
  fig, ax = plt.subplots(1, 1)

  sns.distplot(df2["n=10"][k], hist=True, color="red", kde=True) #edit
  plt.axvline(x=df2["n=10"][k].mean(axis = 0), linestyle='--', color='k')

 # ax.plot(x, expon.pdf(x, scale=1/lambda_), label="Densidad de probabilidad")
 # count, bins, ignored = ax.hist(df2["n=10"][k], int(round(1 + 3.3 * np.log(1000), 0)),rwidth=0.9, color = "green", density = True)
  plt.title('n=10')
  plt.xlabel('x')
  plt.grid(axis='y', alpha=0.75)
  print("La media es: ",df2["n=10"][k].mean(axis = 0))
  print("El coeficiente de asimetria es: ",skew(df2["n=10"][k]))
  plt.show

"""**Acumulativo de todos los datos aleatorios correspondiente a una muestra de tamaño 10 con distribucion exponencial.**"""

dft_exp10 = pd.DataFrame({"n=10":[]})
for n in range (0, 100):
  data = df2.iloc[n, 0]
  for k in range (0, 10):
      nuevo_registro = {"n=10":data[k]}
      dft_exp10=dft_exp10.append(nuevo_registro, ignore_index=True)
dft_exp10
x = dft_exp10
#EXPORTAR EN CSV
#dft_exp10.to_csv("expon_n10.csv")

for n in range (0, 100):
  print("'n"+str(n)+"':[],", end="")

dft_exp1 = pd.DataFrame({'n0':[],'n1':[],'n2':[],'n3':[],'n4':[],'n5':[],'n6':[],'n7':[],'n8':[],'n9':[],'n10':[],'n11':[],'n12':[],'n13':[],'n14':[],'n15':[],'n16':[],'n17':[],'n18':[],'n19':[],'n20':[],'n21':[],'n22':[],'n23':[],'n24':[],'n25':[],'n26':[],'n27':[],'n28':[],'n29':[],'n30':[],'n31':[],'n32':[],'n33':[],'n34':[],'n35':[],'n36':[],'n37':[],'n38':[],'n39':[],'n40':[],'n41':[],'n42':[],'n43':[],'n44':[],'n45':[],'n46':[],'n47':[],'n48':[],'n49':[],'n50':[],'n51':[],'n52':[],'n53':[],'n54':[],'n55':[],'n56':[],'n57':[],'n58':[],'n59':[],'n60':[],'n61':[],'n62':[],'n63':[],'n64':[],'n65':[],'n66':[],'n67':[],'n68':[],'n69':[],'n70':[],'n71':[],'n72':[],'n73':[],'n74':[],'n75':[],'n76':[],'n77':[],'n78':[],'n79':[],'n80':[],'n81':[],'n82':[],'n83':[],'n84':[],'n85':[],'n86':[],'n87':[],'n88':[],'n89':[],'n90':[],'n91':[],'n92':[],'n93':[],'n94':[],'n95':[],'n96':[],'n97':[],'n98':[],'n99':[]})
for n in range (0, 100):
  data = df2.iloc[n, 0]
  dft_exp1["n"+str(n)] = data
dft_exp1
x = dft_exp1
#EXPORTAR EN CSV
dft_exp1.to_csv("expon_n10.csv")

fig, ax = plt.subplots(1, 1)
lambda_ = 10
min_ = 0
max_ = expon.ppf(0.999, scale=1/lambda_)
gran = 100
x = np.linspace(min_, max_, gran)

plt.axvline(dft_exp10["n=10"].mean(axis = 0), linestyle='--', color='k', label="promedio")
ax.plot(x, expon.pdf(x, scale=1/lambda_), label="Densidad de probabilidad")
count, bins, ignored = ax.hist(dft_exp10["n=10"], int(round(1 + 3.3 * np.log(1000), 0)),rwidth=0.9, density = True)
plt.title('n=10')
plt.xlabel('x')
plt.grid(axis='y', alpha=0.75)
print("La media es: ",dft_exp10["n=10"].mean(axis = 0))
print("El coeficiente de asimetria es: ",skew(dft_exp10["n=10"]))
print("curtosis ",  dft_exp10["n=10"].kurt())
print("Varianza ", var(dft_exp10["n=10"]))
plt.show

"""**Histograma de frecuencias relativas para un n=20**"""

lambda_ = 10
min_ = 0
max_ = expon.ppf(0.999, scale=1/lambda_)
gran = 100
x = np.linspace(min_, max_, gran)
for k in range (0,3):
  fig, ax = plt.subplots(1, 1)

  sns.distplot(df2["n=20"][k], hist=True, color="red", kde=True) #edit
  plt.axvline(x=df2["n=20"][k].mean(axis = 0), linestyle='--', color='r')

  #ax.plot(x, expon.pdf(x, scale=1/lambda_), label="Densidad de probabilidad")
 # count, bins, ignored = ax.hist(df2["n=20"][k], int(round(1 + 3.3 * np.log(1000), 0)),rwidth=0.9, color = "red", density = True)
  plt.title('n=20')
  plt.xlabel('x')
  plt.grid(axis='y', alpha=0.75)
  print("La media es: ",df2["n=20"][k].mean(axis = 0))
  print("El coeficiente de asimetria es: ",skew(df2["n=20"][k]))
  plt.show

dft_exp20 = pd.DataFrame({"n=20":[]})
for n in range (0, 100):
  data_2 = df2.iloc[n, 1]
  for k in range (0, 20):
      nuevo_registro = {"n=20":data_2[k]}
      dft_exp20=dft_exp20.append(nuevo_registro, ignore_index=True)
dft_exp20
x2 = dft_exp20
#sns.distplot(x2, hist=False)

#EXPORTAR EN CSV
#dft_exp20.to_csv("expon_n20.csv")

dft_exp2 = pd.DataFrame({'n0':[],'n1':[],'n2':[],'n3':[],'n4':[],'n5':[],'n6':[],'n7':[],'n8':[],'n9':[],'n10':[],'n11':[],'n12':[],'n13':[],'n14':[],'n15':[],'n16':[],'n17':[],'n18':[],'n19':[],'n20':[],'n21':[],'n22':[],'n23':[],'n24':[],'n25':[],'n26':[],'n27':[],'n28':[],'n29':[],'n30':[],'n31':[],'n32':[],'n33':[],'n34':[],'n35':[],'n36':[],'n37':[],'n38':[],'n39':[],'n40':[],'n41':[],'n42':[],'n43':[],'n44':[],'n45':[],'n46':[],'n47':[],'n48':[],'n49':[],'n50':[],'n51':[],'n52':[],'n53':[],'n54':[],'n55':[],'n56':[],'n57':[],'n58':[],'n59':[],'n60':[],'n61':[],'n62':[],'n63':[],'n64':[],'n65':[],'n66':[],'n67':[],'n68':[],'n69':[],'n70':[],'n71':[],'n72':[],'n73':[],'n74':[],'n75':[],'n76':[],'n77':[],'n78':[],'n79':[],'n80':[],'n81':[],'n82':[],'n83':[],'n84':[],'n85':[],'n86':[],'n87':[],'n88':[],'n89':[],'n90':[],'n91':[],'n92':[],'n93':[],'n94':[],'n95':[],'n96':[],'n97':[],'n98':[],'n99':[]})
for n in range (0, 100):
  data_2 = df2.iloc[n, 1]
  dft_exp2["n"+str(n)] = data_2
dft_exp2
x = dft_exp2
#EXPORTAR EN CSV
dft_exp2.to_csv("expon_n20.csv")

"""**Acumulativo de todos los datos aleatorios correspondiente a una muestra de tamaño 20 con distribucion exponencial.**"""

fig, ax = plt.subplots(1, 1)
lambda_ = 10
min_ = 0
max_ = expon.ppf(0.999, scale=1/lambda_)
gran = 100
x = np.linspace(min_, max_, gran)
ax.plot(x, expon.pdf(x, scale=1/lambda_), label="Densidad de probabilidad")
plt.axvline(x=dft_exp20["n=20"].mean(axis = 0), linestyle='--', color='k')
count, bins, ignored = ax.hist(dft_exp20["n=20"], int(round(1 + 3.3 * np.log(2000), 0)),rwidth=0.9, density = True)
plt.title('n=20')
plt.xlabel('x')
plt.grid(axis='y', alpha=0.75)
print("La media es: ",dft_exp20["n=20"].mean(axis = 0))
print("El coeficiente de asimetria es: ",skew(dft_exp20["n=20"]))
print("curtosis ",  dft_exp20["n=20"].kurt())
print("Varianza ", var( dft_exp20["n=20"]))
plt.show

"""**Histograma de frecuencias relativas para un n=50**"""

lambda_ = 10
min_ = 0
max_ = expon.ppf(0.999, scale=1/lambda_)
gran = 100
x = np.linspace(min_, max_, gran)
for k in range (0,3):
  fig, ax = plt.subplots(1, 1)


  sns.distplot(df2["n=50"][k], hist=True, color="red", kde=True) #edit
  plt.axvline(x=df2["n=50"][k].mean(axis = 0), linestyle='--', color='r')

  #ax.plot(x, expon.pdf(x, scale=1/lambda_), label="Densidad de probabilidad")
  #count, bins, ignored = ax.hist(df2["n=50"][k], int(round(1 + 3.3 * np.log(1000), 0)),rwidth=0.9, color = "blue", density = True)
  plt.title('n=50')
  plt.xlabel('x')
  plt.grid(axis='y', alpha=0.75)
  print("La media es: ",df2["n=50"][k].mean(axis = 0))
  print("El coeficiente de asimetria es: ",skew(df2["n=50"][k]))
  plt.show

"""**Acumulativo de todos los datos aleatorios correspondiente a una muestra de tamaño 50 con distribucion exponencial.**"""

dft_exp50 = pd.DataFrame({"n=50":[]})
for n in range (0, 100):
  data_3 = df2.iloc[n, 2]
  for k in range (0, 50):
      nuevo_registro = {"n=50":data_3[k]}
      dft_exp50=dft_exp50.append(nuevo_registro, ignore_index=True)
dft_exp50
x3 = dft_exp50
#sns.distplot(x3, hist=False)
#EXPORTAR EN CSV
#dft_exp50.to_csv("expon_n50.csv")

dft_exp5 = pd.DataFrame({'n0':[],'n1':[],'n2':[],'n3':[],'n4':[],'n5':[],'n6':[],'n7':[],'n8':[],'n9':[],'n10':[],'n11':[],'n12':[],'n13':[],'n14':[],'n15':[],'n16':[],'n17':[],'n18':[],'n19':[],'n20':[],'n21':[],'n22':[],'n23':[],'n24':[],'n25':[],'n26':[],'n27':[],'n28':[],'n29':[],'n30':[],'n31':[],'n32':[],'n33':[],'n34':[],'n35':[],'n36':[],'n37':[],'n38':[],'n39':[],'n40':[],'n41':[],'n42':[],'n43':[],'n44':[],'n45':[],'n46':[],'n47':[],'n48':[],'n49':[],'n50':[],'n51':[],'n52':[],'n53':[],'n54':[],'n55':[],'n56':[],'n57':[],'n58':[],'n59':[],'n60':[],'n61':[],'n62':[],'n63':[],'n64':[],'n65':[],'n66':[],'n67':[],'n68':[],'n69':[],'n70':[],'n71':[],'n72':[],'n73':[],'n74':[],'n75':[],'n76':[],'n77':[],'n78':[],'n79':[],'n80':[],'n81':[],'n82':[],'n83':[],'n84':[],'n85':[],'n86':[],'n87':[],'n88':[],'n89':[],'n90':[],'n91':[],'n92':[],'n93':[],'n94':[],'n95':[],'n96':[],'n97':[],'n98':[],'n99':[]})
for n in range (0, 100):
  data = df2.iloc[n, 2]
  dft_exp5["n"+str(n)] = data
dft_exp5
x = dft_exp5
#EXPORTAR EN CSV
dft_exp5.to_csv("expon_n50.csv")

fig, ax = plt.subplots(1, 1)
lambda_ = 10
min_ = 0
max_ = expon.ppf(0.999, scale=1/lambda_)
gran = 100
x = np.linspace(min_, max_, gran)
ax.plot(x, expon.pdf(x, scale=1/lambda_), label="Densidad de probabilidad")
plt.axvline(x=dft_exp50["n=50"].mean(axis = 0), linestyle='--', color='k')
count, bins, ignored = ax.hist(dft_exp50["n=50"], int(round(1 + 3.3 * np.log(5000), 0)),rwidth=0.9, density = True)
plt.title('n=50')
plt.xlabel('x')
plt.grid(axis='y', alpha=0.75)
print("La media es: ",dft_exp50["n=50"].mean(axis = 0))
print("El coeficiente de asimetria es: ",skew(dft_exp50["n=50"]))
print("curtosis ",  dft_exp50["n=50"].kurt())
print("Varianza ", var( dft_exp50["n=50"]))
plt.show

"""# **Punto 2: Generador de 100 muestras de tamaño 10, 20, 50 con distribucion uniforme con a=5, b=15**"""

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform

#PUNTO 2
dfn1 = pd.DataFrame({"n=10":[], "n=20":[], "n=50":[]})
a=5
b=10
for k in range (0,100):
  n11 = uniform.rvs(a, b, size = 10)
  n22 = uniform.rvs(a, b, size = 20)
  n33 = uniform.rvs(a, b, size = 50)
  nuevo_registro = {"n=10":n11,"n=20":n22,"n=50":n33}
  dfn1=dfn1.append(nuevo_registro, ignore_index=True)
#dfn1
#EXPORTAR DATOS CON LAS 8000 FILAS
dfn1.to_csv("Datos_uniforme.csv")

"""**Histograma de frecuencias relativas para un n=10**"""

for k in range (0,3):
  fig, ax = plt.subplots(1, 1)
  plt.axhline(y=0.1, color='r')
  sns.distplot(dfn1["n=10"][k], hist=False)
  count, bins, ignored = ax.hist(dfn1["n=10"][k], rwidth=0.9, color = "green", density = True)
  plt.title('n=10')
  plt.xlabel('x')
  plt.grid(axis='y', alpha=0.75)
  print("La media es: ",dfn1["n=10"][k].mean(axis = 0))
  plt.show

"""**Acumulativo de todos los datos aleatorios correspondiente a una muestra de tamaño 10 con distribucion uniforme.**"""

dft_unf10 = pd.DataFrame({"n=10":[]})
for n in range (0, 100):
  data = dfn1.iloc[n, 0]
  for k in range (0, 10):
      nuevo_registro = {"n=10":data[k]}
      dft_unf10=dft_unf10.append(nuevo_registro, ignore_index=True)

#EXPORTAR EN CSV
#dft_unf10.to_csv("uniform_n10.csv")

dft_unf1 = pd.DataFrame({'n0':[],'n1':[],'n2':[],'n3':[],'n4':[],'n5':[],'n6':[],'n7':[],'n8':[],'n9':[],'n10':[],'n11':[],'n12':[],'n13':[],'n14':[],'n15':[],'n16':[],'n17':[],'n18':[],'n19':[],'n20':[],'n21':[],'n22':[],'n23':[],'n24':[],'n25':[],'n26':[],'n27':[],'n28':[],'n29':[],'n30':[],'n31':[],'n32':[],'n33':[],'n34':[],'n35':[],'n36':[],'n37':[],'n38':[],'n39':[],'n40':[],'n41':[],'n42':[],'n43':[],'n44':[],'n45':[],'n46':[],'n47':[],'n48':[],'n49':[],'n50':[],'n51':[],'n52':[],'n53':[],'n54':[],'n55':[],'n56':[],'n57':[],'n58':[],'n59':[],'n60':[],'n61':[],'n62':[],'n63':[],'n64':[],'n65':[],'n66':[],'n67':[],'n68':[],'n69':[],'n70':[],'n71':[],'n72':[],'n73':[],'n74':[],'n75':[],'n76':[],'n77':[],'n78':[],'n79':[],'n80':[],'n81':[],'n82':[],'n83':[],'n84':[],'n85':[],'n86':[],'n87':[],'n88':[],'n89':[],'n90':[],'n91':[],'n92':[],'n93':[],'n94':[],'n95':[],'n96':[],'n97':[],'n98':[],'n99':[]})
for n in range (0, 100):
  data = dfn1.iloc[n, 0]
  dft_unf1["n"+str(n)] = data
dft_unf1
x = dft_unf1
#EXPORTAR EN CSV
dft_unf1.to_csv("uniform_n10.csv")

fig, ax = plt.subplots(1, 1)
plt.axhline(y=0.1, color='r')
sns.distplot(dft_unf10["n=10"], hist=False)
count, bins, ignored = ax.hist(dft_unf10["n=10"], rwidth=0.9, density = True)
plt.title('n=10')
plt.xlabel('x')
plt.grid(axis='y', alpha=0.75)
print("La media es: ",dft_unf10["n=10"].mean(axis = 0))
print("Varianza ", var(dft_unf10["n=10"]))
plt.show

"""**Histograma de frecuencias relativas para un n=20**"""

for k in range (0,3):
  fig, ax = plt.subplots(1, 1)
  plt.axhline(y=0.1, color='r')
  sns.distplot(dfn1["n=20"][k], hist=False)
  count, bins, ignored = ax.hist(dfn1["n=20"][k], rwidth=0.9, color = "red", density = True)
  plt.title('n=20')
  plt.xlabel('x')
  plt.grid(axis='y', alpha=0.75)
  print("La media es: ",dfn1["n=20"][k].mean(axis = 0))
  plt.show

"""**Acumulativo de todos los datos aleatorios correspondiente a una muestra de tamaño 20 con distribucion uniforme.**"""

dft_unf20 = pd.DataFrame({"n=20":[]})
for n in range (0, 100):
  data = dfn1.iloc[n, 1]
  for k in range (0, 20):
      nuevo_registro = {"n=20":data[k]}
      dft_unf20=dft_unf20.append(nuevo_registro, ignore_index=True)
#EXPORTAR EN CSV
#dft_unf20.to_csv("uniform_n20.csv")

dft_unf2 = pd.DataFrame({'n0':[],'n1':[],'n2':[],'n3':[],'n4':[],'n5':[],'n6':[],'n7':[],'n8':[],'n9':[],'n10':[],'n11':[],'n12':[],'n13':[],'n14':[],'n15':[],'n16':[],'n17':[],'n18':[],'n19':[],'n20':[],'n21':[],'n22':[],'n23':[],'n24':[],'n25':[],'n26':[],'n27':[],'n28':[],'n29':[],'n30':[],'n31':[],'n32':[],'n33':[],'n34':[],'n35':[],'n36':[],'n37':[],'n38':[],'n39':[],'n40':[],'n41':[],'n42':[],'n43':[],'n44':[],'n45':[],'n46':[],'n47':[],'n48':[],'n49':[],'n50':[],'n51':[],'n52':[],'n53':[],'n54':[],'n55':[],'n56':[],'n57':[],'n58':[],'n59':[],'n60':[],'n61':[],'n62':[],'n63':[],'n64':[],'n65':[],'n66':[],'n67':[],'n68':[],'n69':[],'n70':[],'n71':[],'n72':[],'n73':[],'n74':[],'n75':[],'n76':[],'n77':[],'n78':[],'n79':[],'n80':[],'n81':[],'n82':[],'n83':[],'n84':[],'n85':[],'n86':[],'n87':[],'n88':[],'n89':[],'n90':[],'n91':[],'n92':[],'n93':[],'n94':[],'n95':[],'n96':[],'n97':[],'n98':[],'n99':[]})
for n in range (0, 100):
  data = dfn1.iloc[n, 1]
  dft_unf2["n"+str(n)] = data
dft_unf2
x = dft_unf2
#EXPORTAR EN CSV
dft_unf2.to_csv("uniform_n20.csv")

fig, ax = plt.subplots(1, 1)
plt.axhline(y=0.1, color='r')
sns.distplot(dft_unf20["n=20"], hist=False)
count, bins, ignored = ax.hist(dft_unf20["n=20"],rwidth=0.9, density = True)
plt.title('n=20')
plt.xlabel('x')
plt.grid(axis='y', alpha=0.75)
print("La media es: ",dft_unf20["n=20"].mean(axis = 0))
print("Varianza ", var(dft_unf20["n=20"]))
plt.show

"""**Histograma de frecuencias relativas para un n=50**


"""

for k in range (0,3):
  fig, ax = plt.subplots(1, 1)
  plt.axhline(y=0.1, color='r')
  sns.distplot(dfn1["n=50"][k], hist=False)
  count, bins, ignored = ax.hist(dfn1["n=50"][k], rwidth=0.9, color = "blue", density = True)
  plt.title('n=50')
  plt.xlabel('x')
  plt.grid(axis='y', alpha=0.75)
  print("La media es: ",dfn1["n=50"][k].mean(axis = 0))
  plt.show

"""**Acumulativo de todos los datos aleatorios correspondiente a una muestra de tamaño 50 con distribucion uniforme.**"""

dft_unf50 = pd.DataFrame({"n=50":[]})
for n in range (0, 100):
  data = dfn1.iloc[n, 2]
  for k in range (0, 50):
      nuevo_registro = {"n=50":data[k]}
      dft_unf50=dft_unf50.append(nuevo_registro, ignore_index=True)
#EXPORTAR EN CSV
#dft_unf50.to_csv("uniform_n50.csv")

dft_unf5 = pd.DataFrame({'n0':[],'n1':[],'n2':[],'n3':[],'n4':[],'n5':[],'n6':[],'n7':[],'n8':[],'n9':[],'n10':[],'n11':[],'n12':[],'n13':[],'n14':[],'n15':[],'n16':[],'n17':[],'n18':[],'n19':[],'n20':[],'n21':[],'n22':[],'n23':[],'n24':[],'n25':[],'n26':[],'n27':[],'n28':[],'n29':[],'n30':[],'n31':[],'n32':[],'n33':[],'n34':[],'n35':[],'n36':[],'n37':[],'n38':[],'n39':[],'n40':[],'n41':[],'n42':[],'n43':[],'n44':[],'n45':[],'n46':[],'n47':[],'n48':[],'n49':[],'n50':[],'n51':[],'n52':[],'n53':[],'n54':[],'n55':[],'n56':[],'n57':[],'n58':[],'n59':[],'n60':[],'n61':[],'n62':[],'n63':[],'n64':[],'n65':[],'n66':[],'n67':[],'n68':[],'n69':[],'n70':[],'n71':[],'n72':[],'n73':[],'n74':[],'n75':[],'n76':[],'n77':[],'n78':[],'n79':[],'n80':[],'n81':[],'n82':[],'n83':[],'n84':[],'n85':[],'n86':[],'n87':[],'n88':[],'n89':[],'n90':[],'n91':[],'n92':[],'n93':[],'n94':[],'n95':[],'n96':[],'n97':[],'n98':[],'n99':[]})
for n in range (0, 100):
  data = dfn1.iloc[n, 2]
  dft_unf5["n"+str(n)] = data
dft_unf5
x = dft_unf5
#EXPORTAR EN CSV
dft_unf5.to_csv("uniform_n50.csv")

fig, ax = plt.subplots(1, 1)
plt.axhline(y=0.1, color='r')
sns.distplot(dft_unf50["n=50"], hist=False)
count, bins, ignored = ax.hist(dft_unf50["n=50"],rwidth=0.9, density = True)
plt.title('n=50')
plt.xlabel('x')
plt.grid(axis='y', alpha=0.75)
print("La media es: ",dft_unf50["n=50"].mean(axis = 0))
print("Varianza ", var(dft_unf50["n=50"]))

plt.show

"""# **Punto 3: Generador de 100 muestras de tamaño 10, 20, 50 con distribucion normal**



"""

#PUNTO 3
dfn = pd.DataFrame({"n=10":[], "n=20":[], "n=50":[]})
mu = 10
sigma =2.5
min_ = norm.ppf(0.000001, scale=sigma, loc=mu)
max_ = expon.ppf(0.999, scale=sigma, loc=mu)
gran = 100
x = np.linspace(min_, max_, gran)

for k in range (0,100):
  num_clases7 = int(round(1 + 3.3 * np.log(10), 0))
  n1 = norm.rvs(scale=sigma, loc=mu, size=10)
  if (k<3):
    df2_1 = pd.DataFrame({"n=10":[]})
    for j in range (0, n1.size):
      nuevo_registro = {"n=10":n1[j]}
      df2_1=df2_1.append(nuevo_registro, ignore_index=True)
    print("curtosis grafica ",k+1, "funcion normal ", df2_1.kurt())

  num_clases8 = int(round(1 + 3.3 * np.log(20), 0))
  n2 = norm.rvs(scale=sigma, loc=mu, size=20)
  if (k<3):
    df2_1 = pd.DataFrame({"n=20":[]})
    for j in range (0, n2.size):
      nuevo_registro = {"n=20":n2[j]}
      df2_1=df2_1.append(nuevo_registro, ignore_index=True)
    print("curtosis grafica ",k+1, "funcion normal ", df2_1.kurt())

  num_clases9 = int(round(1 + 3.3 * np.log(50), 0))
  n3 = norm.rvs(scale=sigma, loc=mu, size=50)
  if (k<3):
    df2_1 = pd.DataFrame({"n=50":[]})
    for j in range (0, n3.size):
      nuevo_registro = {"n=50":n3[j]}
      df2_1=df2_1.append(nuevo_registro, ignore_index=True)
    print("curtosis grafica ",k+1, "funcion normal ", df2_1.kurt())

  nuevo_registro = {"n=10":n1,"n=20":n2,"n=50":n3}

  dfn=dfn.append(nuevo_registro, ignore_index=True)

#EXPORTAR DATOS CON LAS 8000 FILAS
dfn.to_csv("Datos_normal.csv")

"""**Histograma de frecuencias relativas para un n=10**"""

for k in range (0,3):
  fig, ax = plt.subplots(1, 1)
  min_ = norm.ppf(0.000001, scale=sigma, loc=mu)
  max_ = expon.ppf(0.999, scale=sigma, loc=mu)
  gran = 100
  x = np.linspace(min_, max_, gran)
  plt.axvline(x=dfn["n=10"][k].mean(axis = 0), linestyle='--', color='k')
  ax.plot(x, norm.pdf(x, scale=sigma, loc=mu), label="Densidad de probabilidad")
  count, bins, ignored = ax.hist(dfn["n=10"][k], rwidth=0.9, color="green", density = True)
  plt.title('n=10')
  plt.xlabel('x')
  plt.grid(axis='y', alpha=0.75)
  print("La media es: ",dfn["n=10"][k].mean(axis = 0))
  print("El coeficiente de asimetria es: ",skew(dfn["n=10"][k]))
  plt.show

"""**Acumulativo de todos los datos aleatorios correspondiente a una muestra de tamaño 10 con distribucion normal.**"""

dft_norm10 = pd.DataFrame({"n=10":[]})
for n in range (0, 100):
  data = dfn.iloc[n, 0]
  for k in range (0, 10):
      nuevo_registro = {"n=10":data[k]}
      dft_norm10=dft_norm10.append(nuevo_registro, ignore_index=True)
#EXPORTAR EN CSV
#dft_norm10.to_csv("norm_n10.csv")

dft_norm1 = pd.DataFrame({'n0':[],'n1':[],'n2':[],'n3':[],'n4':[],'n5':[],'n6':[],'n7':[],'n8':[],'n9':[],'n10':[],'n11':[],'n12':[],'n13':[],'n14':[],'n15':[],'n16':[],'n17':[],'n18':[],'n19':[],'n20':[],'n21':[],'n22':[],'n23':[],'n24':[],'n25':[],'n26':[],'n27':[],'n28':[],'n29':[],'n30':[],'n31':[],'n32':[],'n33':[],'n34':[],'n35':[],'n36':[],'n37':[],'n38':[],'n39':[],'n40':[],'n41':[],'n42':[],'n43':[],'n44':[],'n45':[],'n46':[],'n47':[],'n48':[],'n49':[],'n50':[],'n51':[],'n52':[],'n53':[],'n54':[],'n55':[],'n56':[],'n57':[],'n58':[],'n59':[],'n60':[],'n61':[],'n62':[],'n63':[],'n64':[],'n65':[],'n66':[],'n67':[],'n68':[],'n69':[],'n70':[],'n71':[],'n72':[],'n73':[],'n74':[],'n75':[],'n76':[],'n77':[],'n78':[],'n79':[],'n80':[],'n81':[],'n82':[],'n83':[],'n84':[],'n85':[],'n86':[],'n87':[],'n88':[],'n89':[],'n90':[],'n91':[],'n92':[],'n93':[],'n94':[],'n95':[],'n96':[],'n97':[],'n98':[],'n99':[]})
for n in range (0, 100):
  data = dfn.iloc[n, 0]
  dft_norm1["n"+str(n)] = data
dft_norm1
x = dft_norm1
#EXPORTAR EN CSV
dft_norm1.to_csv("norm_n10.csv")

fig, ax = plt.subplots(1, 1)
min_ = norm.ppf(0.000001, scale=sigma, loc=mu)
max_ = expon.ppf(0.999, scale=sigma, loc=mu)
gran = 100
x = np.linspace(min_, max_, gran)
plt.axvline(x=dft_norm10["n=10"].mean(axis = 0), linestyle='--', color='k')
ax.plot(x, norm.pdf(x, scale=sigma, loc=mu), label="Densidad de probabilidad")
count, bins, ignored = ax.hist(dft_norm10["n=10"], rwidth=0.9, density = True)
plt.title('n=10')
plt.xlabel('x')
plt.grid(axis='y', alpha=0.75)
print("La media es: ",dft_norm10["n=10"].mean(axis = 0))
print("El coeficiente de asimetria es: ",skew(dft_norm10["n=10"]))
print("curtosis: ",dft_norm10["n=10"].kurt())
print("Varianza ", var(dft_norm10["n=10"]))
plt.show

"""**Histograma de frecuencias relativas para un n=20**"""

for k in range (0,3):
  fig, ax = plt.subplots(1, 1)
  min_ = norm.ppf(0.000001, scale=sigma, loc=mu)
  max_ = expon.ppf(0.999, scale=sigma, loc=mu)
  gran = 100
  x = np.linspace(min_, max_, gran)
  plt.axvline(x=dfn["n=20"][k].mean(axis = 0), linestyle='--', color='k')

  ax.plot(x, norm.pdf(x, scale=sigma, loc=mu), label="Densidad de probabilidad")
  count, bins, ignored = ax.hist(dfn["n=20"][k], rwidth=0.9, color="red", density = True)
  plt.title('n=20')
  plt.xlabel('x')
  plt.grid(axis='y', alpha=0.75)
  print("La media es: ",dfn["n=20"][k].mean(axis = 0))
  print("El coeficiente de asimetria es: ",skew(dfn["n=20"][k]))
  plt.show

"""**Acumulativo de todos los datos aleatorios correspondiente a una muestra de tamaño 20 con distribucion normal.**"""

dft_norm20 = pd.DataFrame({"n=20":[]})
for n in range (0, 100):
  data = dfn.iloc[n, 1]
  for k in range (0, 20):
      nuevo_registro = {"n=20":data[k]}
      dft_norm20=dft_norm20.append(nuevo_registro, ignore_index=True)
#EXPORTAR EN CSV
#dft_norm20.to_csv("norm_n20.csv")

dft_norm2 = pd.DataFrame({'n0':[],'n1':[],'n2':[],'n3':[],'n4':[],'n5':[],'n6':[],'n7':[],'n8':[],'n9':[],'n10':[],'n11':[],'n12':[],'n13':[],'n14':[],'n15':[],'n16':[],'n17':[],'n18':[],'n19':[],'n20':[],'n21':[],'n22':[],'n23':[],'n24':[],'n25':[],'n26':[],'n27':[],'n28':[],'n29':[],'n30':[],'n31':[],'n32':[],'n33':[],'n34':[],'n35':[],'n36':[],'n37':[],'n38':[],'n39':[],'n40':[],'n41':[],'n42':[],'n43':[],'n44':[],'n45':[],'n46':[],'n47':[],'n48':[],'n49':[],'n50':[],'n51':[],'n52':[],'n53':[],'n54':[],'n55':[],'n56':[],'n57':[],'n58':[],'n59':[],'n60':[],'n61':[],'n62':[],'n63':[],'n64':[],'n65':[],'n66':[],'n67':[],'n68':[],'n69':[],'n70':[],'n71':[],'n72':[],'n73':[],'n74':[],'n75':[],'n76':[],'n77':[],'n78':[],'n79':[],'n80':[],'n81':[],'n82':[],'n83':[],'n84':[],'n85':[],'n86':[],'n87':[],'n88':[],'n89':[],'n90':[],'n91':[],'n92':[],'n93':[],'n94':[],'n95':[],'n96':[],'n97':[],'n98':[],'n99':[]})
for n in range (0, 100):
  data = dfn.iloc[n, 1]
  dft_norm2["n"+str(n)] = data
dft_norm2
x = dft_norm2
#EXPORTAR EN CSV
dft_norm2.to_csv("norm_n20.csv")

fig, ax = plt.subplots(1, 1)
min_ = norm.ppf(0.000001, scale=sigma, loc=mu)
max_ = expon.ppf(0.999, scale=sigma, loc=mu)
gran = 100
x = np.linspace(min_, max_, gran)

ax.plot(x, norm.pdf(x, scale=sigma, loc=mu), label="Densidad de probabilidad")
count, bins, ignored = ax.hist(dft_norm20["n=20"], rwidth=0.9, density = True)
plt.axvline(x=dft_norm20["n=20"].mean(axis = 0), linestyle='--', color='k')
plt.title('n=20')
plt.xlabel('x')
plt.grid(axis='y', alpha=0.75)
print("La media es: ",dft_norm20["n=20"].mean(axis = 0))
print("El coeficiente de asimetria es: ",skew(dft_norm20["n=20"]))
print("curtosis: ",dft_norm20["n=20"].kurt())
print("Varianza ", var(dft_norm20["n=20"]))
plt.show

"""**Histograma de frecuencias relativas para un n=50**"""

for k in range (0,3):
  fig, ax = plt.subplots(1, 1)
  min_ = norm.ppf(0.000001, scale=sigma, loc=mu)
  max_ = expon.ppf(0.999, scale=sigma, loc=mu)
  gran = 100
  x = np.linspace(min_, max_, gran)
  ax.plot(x, norm.pdf(x, scale=sigma, loc=mu), label="Densidad de probabilidad")
  count, bins, ignored = ax.hist(dfn["n=50"][k], rwidth=0.9, color="blue", density = True)
  plt.axvline(x=dfn["n=50"][k].mean(axis = 0), linestyle='--', color='red')
  plt.title('n=50')
  plt.xlabel('x')
  plt.grid(axis='y', alpha=0.75)
  print("La media es: ",dfn["n=50"][k].mean(axis = 0))
  print("El coeficiente de asimetria es: ",skew(dfn["n=50"][k]))
  plt.show

"""**Acumulativo de todos los datos aleatorios correspondiente a una muestra de tamaño 50 con distribucion normal.**"""

dft_norm50 = pd.DataFrame({"n=50":[]})
for n in range (0, 100):
  data = dfn.iloc[n, 2]
  for k in range (0, 50):
      nuevo_registro = {"n=50":data[k]}
      dft_norm50=dft_norm50.append(nuevo_registro, ignore_index=True)
#EXPORTAR EN CSV
#dft_norm50.to_csv("norm_n50.csv")

dft_norm5 = pd.DataFrame({'n0':[],'n1':[],'n2':[],'n3':[],'n4':[],'n5':[],'n6':[],'n7':[],'n8':[],'n9':[],'n10':[],'n11':[],'n12':[],'n13':[],'n14':[],'n15':[],'n16':[],'n17':[],'n18':[],'n19':[],'n20':[],'n21':[],'n22':[],'n23':[],'n24':[],'n25':[],'n26':[],'n27':[],'n28':[],'n29':[],'n30':[],'n31':[],'n32':[],'n33':[],'n34':[],'n35':[],'n36':[],'n37':[],'n38':[],'n39':[],'n40':[],'n41':[],'n42':[],'n43':[],'n44':[],'n45':[],'n46':[],'n47':[],'n48':[],'n49':[],'n50':[],'n51':[],'n52':[],'n53':[],'n54':[],'n55':[],'n56':[],'n57':[],'n58':[],'n59':[],'n60':[],'n61':[],'n62':[],'n63':[],'n64':[],'n65':[],'n66':[],'n67':[],'n68':[],'n69':[],'n70':[],'n71':[],'n72':[],'n73':[],'n74':[],'n75':[],'n76':[],'n77':[],'n78':[],'n79':[],'n80':[],'n81':[],'n82':[],'n83':[],'n84':[],'n85':[],'n86':[],'n87':[],'n88':[],'n89':[],'n90':[],'n91':[],'n92':[],'n93':[],'n94':[],'n95':[],'n96':[],'n97':[],'n98':[],'n99':[]})
for n in range (0, 100):
  data = dfn.iloc[n, 2]
  dft_norm5["n"+str(n)] = data
dft_norm5
x = dft_norm5
#EXPORTAR EN CSV
dft_norm5.to_csv("norm_n50.csv")

fig, ax = plt.subplots(1, 1)
min_ = norm.ppf(0.000001, scale=sigma, loc=mu)
max_ = expon.ppf(0.999, scale=sigma, loc=mu)
gran = 100
x = np.linspace(min_, max_, gran)
ax.plot(x, norm.pdf(x, scale=sigma, loc=mu), label="Densidad de probabilidad")
count, bins, ignored = ax.hist(dft_norm50["n=50"], rwidth=0.9, density = True)
plt.axvline(x=dft_norm50["n=50"].mean(axis = 0), linestyle='--', color='k')
plt.title('n=50')
plt.xlabel('x')
plt.grid(axis='y', alpha=0.75)
print("La media es: ",dft_norm50["n=50"].mean(axis = 0))
print("El coeficiente de asimetria es: ",skew(dft_norm50["n=50"]))
print("curtosis: ",dft_norm50["n=50"].kurt())
print("Varianza ", var(dft_norm50["n=50"]))
plt.show

"""# PUNTO 8"""

dfpex = pd.DataFrame({"Promedios n=10":[], "Promedios n=20":[], "Promedios n=50":[]})
dfpun = pd.DataFrame({"Promedios n=10":[], "Promedios n=20":[], "Promedios n=50":[]})
dfpnor = pd.DataFrame({"Promedios n=10":[], "Promedios n=20":[], "Promedios n=50":[]})
for k in range (0, 100):
  nuevo_registro1 = {"Promedios n=10":df2["n=10"][k].mean(axis = 0), "Promedios n=20":df2["n=20"][k].mean(axis = 0), "Promedios n=50":df2["n=50"][k].mean(axis = 0)}
  dfpex=dfpex.append(nuevo_registro1, ignore_index=True)
  nuevo_registro2 = {"Promedios n=10":dfn1["n=10"][k].mean(axis = 0), "Promedios n=20":dfn1["n=20"][k].mean(axis = 0), "Promedios n=50":dfn1["n=50"][k].mean(axis = 0)}
  dfpun=dfpun.append(nuevo_registro2, ignore_index=True)
  nuevo_registro3 = {"Promedios n=10":dfn["n=10"][k].mean(axis = 0), "Promedios n=20":dfn["n=20"][k].mean(axis = 0), "Promedios n=50":dfn["n=50"][k].mean(axis = 0)}
  dfpnor=dfpnor.append(nuevo_registro3, ignore_index=True)

print("Distribucion de medias muestrales provenientes de una distribucion exponencial")
print("n=10")
print("Media: ", dfpex["Promedios n=10"].mean(), " Varianza: ",var(dfpex["Promedios n=10"]))
print("n=20")
print("Media: ", dfpex["Promedios n=20"].mean(), " Varianza: ",var(dfpex["Promedios n=20"]))
print("n=50")
print("Media: ", dfpex["Promedios n=50"].mean(), " Varianza: ",var(dfpex["Promedios n=50"]))
print(" ")
print("Distribucion de medias muestrales provenientes de una distribucion uniforme")
print("n=10")
print("Media: ", dfpun["Promedios n=10"].mean(), " Varianza: ",var(dfpun["Promedios n=10"]))
print("n=20")
print("Media: ", dfpun["Promedios n=20"].mean(), " Varianza: ",var(dfpun["Promedios n=20"]))
print("n=50")
print("Media: ", dfpun["Promedios n=50"].mean(), " Varianza: ",var(dfpun["Promedios n=50"]))
print(" ")
print("Distribucion de medias muestrales provenientes de una distribucion normal")
print("n=10")
print("Media: ", dfpnor["Promedios n=10"].mean(), " Varianza: ",var(dfpnor["Promedios n=10"]))
print("n=20")
print("Media: ", dfpnor["Promedios n=20"].mean(), " Varianza: ",var(dfpnor["Promedios n=20"]))
print("n=50")
print("Media: ", dfpnor["Promedios n=50"].mean(), " Varianza: ",var(dfpnor["Promedios n=50"]))

"""# PUNTO 9, distribuciones de medias muestrales

la fila que contiene las graficas de color azul claro corresponde a las distribuciones de media para muestras con distribucion exponencial para un n=10, 20 y 50 respectivamente.

la fila que contiene las graficas de color verde corresponde a las distribuciones de media para muestras con distribucion uniforme para un n=10, 20 y 50 respectivamente.

la fila que contiene las graficas de color naranja corresponde a las distribuciones de media para muestras con distribucion normal para un n=10, 20 y 50 respectivamente.
"""

import matplotlib.pyplot as plt
num_clases = int(round(1 + 3.3 * np.log(100), 0))
fig, [(ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)] = plt.subplots(nrows = 3, ncols = 3, figsize=(10,8))
ax0.hist(dfpex["Promedios n=10"],bins=num_clases, color="skyblue", density = True)
ax1.hist(dfpex["Promedios n=20"],bins=num_clases, color="skyblue", density = True)
ax2.hist(dfpex["Promedios n=50"],bins=num_clases,color="skyblue",  density = True)
ax3.hist(dfpun["Promedios n=10"],bins=num_clases, color="#42A84F", density = True)
ax4.hist(dfpun["Promedios n=20"],bins=num_clases, color="#42A84F", density = True)
ax5.hist(dfpun["Promedios n=50"],bins=num_clases, color="#42A84F", density = True)
ax6.hist(dfpnor["Promedios n=10"], bins=num_clases, color="darkorange", density = True)
ax7.hist(dfpnor["Promedios n=20"],bins=num_clases,   color="darkorange", density = True)
ax8.hist(dfpnor["Promedios n=50"], bins=num_clases,  color="darkorange", density = True)
plt.show

"""# PUNTO 10, distribucion de medias muestrales vs distribucion de probabilidad

Es lo mismo que para el punto 9, la unica diferencia radica en que se coloca en la misma grafica de distribucion de medias la distribucion de probabilidad, por eso hay dos distribuciones en una grafica.

Las distribuciones de probabilidad originales son las que estan de color azul fuerte.
"""

import matplotlib.pyplot as plt
num_clases = int(round(1 + 3.3 * np.log(100), 0))
fig, [(ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)] = plt.subplots(nrows = 3, ncols = 3, figsize=(16,12))

ax0.hist(dfpex["Promedios n=10"],bins=num_clases, color="skyblue", density = True)
ax0.hist(dft_exp10["n=10"], int(round(1 + 3.3 * np.log(1000), 0)), density = True)

ax1.hist(dfpex["Promedios n=20"],bins=num_clases, color="skyblue", density = True)
ax1.hist(dft_exp20["n=20"], int(round(1 + 3.3 * np.log(2000), 0)), density = True)

ax2.hist(dfpex["Promedios n=50"],bins=num_clases,color="skyblue",  density = True)
ax2.hist(dft_exp50["n=50"], int(round(1 + 3.3 * np.log(5000), 0)), density = True)

ax3.hist(dfpun["Promedios n=10"],bins=num_clases, color="#42A84F", density = True)
ax3.hist(dft_unf10["n=10"], int(round(1 + 3.3 * np.log(1000), 0)), density = True)

ax4.hist(dfpun["Promedios n=20"],bins=num_clases, color="#42A84F", density = True)
ax4.hist(dft_unf20["n=20"], int(round(1 + 3.3 * np.log(2000), 0)),density = True)

ax5.hist(dfpun["Promedios n=50"],bins=num_clases, color="#42A84F", density = True)
ax5.hist(dft_unf50["n=50"],int(round(1 + 3.3 * np.log(5000), 0)), density = True)

ax6.hist(dfpnor["Promedios n=10"], bins=num_clases, color="darkorange", density = True)
ax6.hist(dft_norm10["n=10"], int(round(1 + 3.3 * np.log(1000), 0)), density = True)

ax7.hist(dfpnor["Promedios n=20"],bins=num_clases,   color="darkorange", density = True)
ax7.hist(dft_norm20["n=20"],int(round(1 + 3.3 * np.log(2000), 0)),  density = True)

ax8.hist(dfpnor["Promedios n=50"], bins=num_clases,  color="darkorange", density = True)
ax8.hist(dft_norm50["n=50"],int(round(1 + 3.3 * np.log(5000), 0)),  density = True)


plt.show

"""# PUNTO 11, pruebas de normalidad"""

from scipy.stats import shapiro
from scipy.stats import norm

"""

> **Distribucion de medias muestrales extraidas de una distribucion exponencial con n=10**

"""

# Prueba Shapiro-Wilk
alpha = 0.05 # se establece el riesgo de la prueba (complemento de la confiabilidad)
x = dfpex["Promedios n=10"]
result = tuple(shapiro(x))
print("p-value = "+str(result[1]))
if result[1] < alpha:
    print("Los datos NO se ajustan a una distribución normal")
else:
    print("Los datos se ajustan a una distribución normal")

"""
> **Distribucion de medias muestrales extraidas de una distribucion exponencial con n=20**"""

# Prueba Shapiro-Wilk
alpha = 0.05 # se establece el riesgo de la prueba (complemento de la confiabilidad)
x = dfpex["Promedios n=20"]
result = tuple(shapiro(x))
print("p-value = "+str(result[1]))
if result[1] < alpha:
    print("Los datos NO se ajustan a una distribución normal")
else:
    print("Los datos se ajustan a una distribución normal")

"""
> **Distribucion de medias muestrales extraidas de una distribucion exponencial con n=50**"""

# Prueba Shapiro-Wilk
alpha = 0.05 # se establece el riesgo de la prueba (complemento de la confiabilidad)
x = dfpex["Promedios n=50"]
result = tuple(shapiro(x))
print("p-value = "+str(result[1]))
if result[1] < alpha:
    print("Los datos NO se ajustan a una distribución normal")
else:
    print("Los datos se ajustan a una distribución normal")

"""
> **Distribucion de medias muestrales extraidas de una distribucion uniforme con n=10**"""

# Prueba Shapiro-Wilk
alpha = 0.05 # se establece el riesgo de la prueba (complemento de la confiabilidad)
x = dfpun["Promedios n=10"]
result = tuple(shapiro(x))
print("p-value = "+str(result[1]))
if result[1] < alpha:
    print("Los datos NO se ajustan a una distribución normal")
else:
    print("Los datos se ajustan a una distribución normal")

"""> **Distribucion de medias muestrales extraidas de una distribucion uniforme con n=20**"""

# Prueba Shapiro-Wilk
alpha = 0.05 # se establece el riesgo de la prueba (complemento de la confiabilidad)
x = dfpun["Promedios n=20"]
result = tuple(shapiro(x))
print("p-value = "+str(result[1]))
if result[1] < alpha:
    print("Los datos NO se ajustan a una distribución normal")
else:
    print("Los datos se ajustan a una distribución normal")

"""> **Distribucion de medias muestrales extraidas de una distribucion uniforme con n=50**"""

# Prueba Shapiro-Wilk
alpha = 0.05 # se establece el riesgo de la prueba (complemento de la confiabilidad)
x = dfpun["Promedios n=50"]
result = tuple(shapiro(x))
print("p-value = "+str(result[1]))
if result[1] < alpha:
    print("Los datos NO se ajustan a una distribución normal")
else:
    print("Los datos se ajustan a una distribución normal")

"""> **Distribucion de medias muestrales extraidas de una distribucion normal con n=10**"""

# Prueba Shapiro-Wilk
alpha = 0.05 # se establece el riesgo de la prueba (complemento de la confiabilidad)
x = dfpnor["Promedios n=10"]
result = tuple(shapiro(x))
print("p-value = "+str(result[1]))
if result[1] < alpha:
    print("Los datos NO se ajustan a una distribución normal")
else:
    print("Los datos se ajustan a una distribución normal")

"""> **Distribucion de medias muestrales extraidas de una distribucion normal con n=20**"""

# Prueba Shapiro-Wilk
alpha = 0.05 # se establece el riesgo de la prueba (complemento de la confiabilidad)
x = dfpnor["Promedios n=20"]
result = tuple(shapiro(x))
print("p-value = "+str(result[1]))
if result[1] < alpha:
    print("Los datos NO se ajustan a una distribución normal")
else:
    print("Los datos se ajustan a una distribución normal")

"""> **Distribucion de medias muestrales extraidas de una distribucion normal con n=50**"""

# Prueba Shapiro-Wilk
alpha = 0.05 # se establece el riesgo de la prueba (complemento de la confiabilidad)
x = dfpnor["Promedios n=50"]
result = tuple(shapiro(x))
print("p-value = "+str(result[1]))
if result[1] < alpha:
    print("Los datos NO se ajustan a una distribución normal")
else:
    print("Los datos se ajustan a una distribución normal")