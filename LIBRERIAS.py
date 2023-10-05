# Paqueterías Básicas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generar Datsets de Trabajo
from sklearn import datasets

# Generar Datos de Entrenamiento y Testeo
from sklearn.model_selection import train_test_split

# Metricas Validación Cruzada
from sklearn.model_selection import cross_val_score
from sklearn import metrics #Metricas Validación Cruzada
from sklearn import svm 
# Estrategias de Validación para CV
from sklearn.model_selection import ShuffleSplit

# Trasformación de datos con datos retenidos
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

# Cross Validate Function and Multiple Metric Evaluation
#Empleando Lista
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score

# Funcion Validacion Cruzada y Evaluación de Multiple Metricas
# Empleando Diccionario
from sklearn.metrics import make_scorer

# Iteradores Validación Cruzada
#K-Fold
from sklearn.model_selection import KFold
#Repeat K-Fold
from sklearn.model_selection import RepeatedKFold

# LeaveOnOut(LOO)
from sklearn.model_selection import LeaveOneOut

#Leave P Out (LPO)
from sklearn.model_selection import LeavePOut

# Validación cruzada de permutaciones aleatorias (Shuffle & Split)
from sklearn.model_selection import ShuffleSplit

# Iteradores de Validación Cruzada con estratificaciones basadas en etiquetas de clases.
from sklearn.model_selection import StratifiedKFold

# Iteradores de Validación Cruzada para un grupo de datos
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeavePGroupsOut

# Uso de iteradores de validación cruzada para dividir datos de entrenamiento y datos de testeo
from sklearn.model_selection import GroupShuffleSplit

# Validación cruzada de datos de series temporales.
from sklearn.model_selection import TimeSeriesSplit

# Calculo de Varianza y Sesgo 

from mlxtend.evaluate import bias_variance_decomp


# Librerías para Regresiones 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

import statsmodels.api as sm  