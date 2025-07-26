# PREDICCIÓN DE DESERCIÓN BANCARIA
<img width="800" height="428" alt="image" src="https://github.com/user-attachments/assets/9bc8a8d8-265a-41cc-aeaf-f6b3f6eb28de" />

## Introducción

La retención de clientes en el sector bancario es un desafío fundamental para la sostenibilidad financiera de cualquier institución. La competencia en la industria, junto con la facilidad con la que los clientes pueden migrar a otras entidades, hace que la fidelización se convierta en una prioridad estratégica. 

Por otro lado, fidelización de clientes individualizada es difícil, la mayoría de las empresas tienen una gran cantidad de clientes y no pueden dedicar mucho tiempo a cada uno. Adicional los costes serían excesivos, superando los ingresos adicionales. Sin embargo, si una empresa pudiera prever qué clientes es probable que abandonen la empresa, podría centrar sus esfuerzos de fidelización únicamente en estos clientes de alto riesgo de deserción.

La pérdida de clientes es una métrica importante,  porque a largo plazo para el banco es mucho más económico y seguro, retener a los clientes existentes que adquirir nuevos.
Este proyecto tiene como objetivo analizar un conjunto de datos bancarios para comprender los factores asociados al abandono de clientes y detectar señales tempranas de una posible pérdida, para desarrollar un modelo predictivo. Para ello, se realiza un análisis estadístico exploratorio detallado de las variables disponibles, enfocándose en aquellas que podrían ayudar a predecir el comportamiento de los clientes más propensos a deserción y de esta forma dedicar un plan de fidelización para estos clientes,  disminuyendo la pérdida de clientes y minimizando los costos de campañas de fidelización.


## Justificación

El costo de adquirir un nuevo cliente bancario suele ser significativamente más alto que el de retener uno existente, además,  un cliente nuevo no significa un cliente permanente y toma tiempo construir una relación de confianza y que la empresa recupere la inversión de captar ese cliente.Por tanto, identificar clientes en riesgo de abandono permite diseñar estrategias proactivas que reduzcan drásticamente la perdida de clientes en la empresa. 
Mediante herramientas estadíticas y modelos predictivos, es posible construir modelos que anticipen esta conducta de abandono que no permita tomar acciones de perder un cliente.
Este proyecto busca sentar las bases para dicha predicción a partir de datos de los clientes, con una exploración de variables que serían utilizadas en el entrenamiento del modelo y levantar un modelo que permita obtimizar los recursos de fidelización y minimize las perdidas de clientes propensos a desertar.



## Antecedentes

Existen múltiples estudios que abordan la predicción del abandono de clientes (customer churn) mediante técnicas de machine learning. 
Por ejemplo, un artículo publicado por IBM presenta el uso de árboles de decisión y modelos de boosting para detectar churn en telecomunicaciones. 
https://www.ibm.com/cloud/learn/customer-churn-prediction

Otro estudio de Towards Data Science aplica redes neuronales y técnicas de reducción de dimensionalidad en bancos para este propósito.
https://towardsdatascience.com/customer-churn-prediction-in-python-d6a5d5f520d

Además, plataformas como Kaggle albergan competencias con datasets similares, mostrando que esta es una problemática ampliamente estudiada con alto impacto.
https://www.kaggle.com/blastchar/telco-customer-churn


## Definición del problema

¿Qué características definen a los clientes con mayor probabilidad de abandonar el banco?

Esta investigación busca identificar y cuantificar esas variables clave a partir de un análisis exploratorio de datos históricos, en preparación para una posterior etapa de modelado predictivo.
Avance del análisis predictivo y Visualizaciones
Hasta este punto se ha realizado una exploración detallada del conjunto de datos. Algunas variables muestran diferencias marcadas entre los clientes que abandonan y los que permanecen, como la edad y el saldo bancario. Los clientes que abandonan tienen en promedio 7 años más y saldos superiores por más de 18 mil euros. Esto sugiere que estas variables podrían ser útiles para predecir el abandono. También se observan diferencias por país y género, aunque de menor magnitud.


## Avance del análisis predictivo y Visualizaciones

Hasta este punto se ha realizado una exploración detallada del conjunto de datos. Algunas variables muestran diferencias marcadas entre los clientes que abandonan y los que permanecen, como la edad y el saldo bancario. Los clientes que abandonan tienen en promedio 7 años más y saldos superiores por más de 18 mil euros. Esto sugiere que estas variables podrían ser útiles para predecir el abandono. También se observan diferencias por país y género, aunque de menor magnitud.

### Estadíticas Generales
<img width="804" height="195" alt="image" src="https://github.com/user-attachments/assets/b69fca40-dd66-4251-8aa0-93e99fd9d848" />

 <img width="724" height="242" alt="image" src="https://github.com/user-attachments/assets/8d1135c2-c203-485a-bb55-503420852505" />

 

### Hipotesis de Variables
Buscamos analizar las variables con el fin de identificar cual de estas varibles puede ayudar al modelo a distinguir una clase de la otra.

Total_Trans_Ct y Total_Trans_Amt: Los clientes que abandonaron (Attrited) suelen tener menos transacciones y menor volumen total.


<img width="402" height="248" alt="image" src="https://github.com/user-attachments/assets/c0163d3f-3e1c-450d-947c-35c6fec0c34d" />

<img width="411" height="254" alt="image" src="https://github.com/user-attachments/assets/3b39d0f7-77f9-4f20-9ff3-a4feda38609f" />


Avg_Utilization_Ratio: Tiende a ser mayor en clientes que se fueron → utilizan más porcentaje de su crédito disponible.


<img width="531" height="350" alt="image" src="https://github.com/user-attachments/assets/091cd904-25cb-4fe0-b78e-8bcaff259f1c" />


Total_Amt_Chng_Q4_Q1 y Total_Ct_Chng_Q4_Q1: Los Attrited tienen cambios más bajos en montos y frecuencia ente trimestres → menor actividad reciente.


<img width="555" height="354" alt="image" src="https://github.com/user-attachments/assets/7f13143e-ee7b-405b-8e41-f3eb690fe9e7" />

<img width="582" height="371" alt="image" src="https://github.com/user-attachments/assets/7e8750f9-4179-4816-a818-707c7e742e37" />

Total_Revolving_Bal por clase de cliente: tiende a ser mas bajo para los clientes Attrited.


<img width="523" height="250" alt="image" src="https://github.com/user-attachments/assets/2825f973-0196-442e-88ae-38d03e10ce61" />



### Target

<img width="645" height="110" alt="image" src="https://github.com/user-attachments/assets/493d57f6-6e90-4ffa-b43e-9584585cc6b8" />



## Entrenamiento de Modelo

### Limpieza de Datos se eliminaron, las columnas:
* df = df.drop(columns=[
    'CLIENTNUM',
'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
])

### Split data 
 
* Se dividió la data en 80/20, reservando 20% para el testeo.
  
### TRAIN

* X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Modelos Entrenados

Se entrenaron 2 modelos de clasificación, RandomForestClassifier y LGBMClassifier  y se compararon sus desempeños.

## RANDOM FOREST
Métricas de Desempeño:
               precision    recall  f1-score   support

           0       0.96      0.99      0.97      1701
           1       0.94      0.77      0.85       325

    accuracy                           0.96      2026
   macro avg       0.95      0.88      0.91      2026
weighted avg       0.95      0.96      0.95      2026

MAE: 0.0449
RMSE: 0.2119
MAPE: 4.49%
AUC ROC: 0.9841

### Matriz de Confusión

<img width="426" height="367" alt="image" src="https://github.com/user-attachments/assets/07374f85-2a02-44eb-8f63-bc9f189f843c" />

### Curva ROC


<img width="509" height="393" alt="image" src="https://github.com/user-attachments/assets/36dbbffc-f371-4294-85a2-abf87ef5d42a" />




## LIGHTGBM
Métricas de Desempeño:
               precision    recall  f1-score   support

           0       0.98      0.99      0.98      1701
           1       0.94      0.89      0.92       325

    accuracy                           0.97      2026
   macro avg       0.96      0.94      0.95      2026
weighted avg       0.97      0.97      0.97      2026

MAE: 0.0262
RMSE: 0.1617
MAPE: 2.62%
AUC ROC: 0.9925

### Matriz de Confusión

<img width="602" height="473" alt="image" src="https://github.com/user-attachments/assets/b572bcfe-6b44-4f17-bf1f-793d3dda0686" />

Curva ROC

<img width="477" height="368" alt="image" src="https://github.com/user-attachments/assets/de6f1d51-f2dd-48ca-a792-a9e1cd39cc91" />


## COMPARATIVA DE MODELOS



<img width="866" height="468" alt="image" src="https://github.com/user-attachments/assets/da4a20c2-e404-4941-906b-f0b0413b0837" />


Si bien ambos modelos son muy buenos modelos y lograr separar las clases de buena manera, el modelo de Lightgbm es mejor que el Random Forest opteniendo mejores métricas y destaca en la captura de mayores casos de deserción. Tambien sus erros son mas bajos, por todo esto el escogeremos el modelo de Lightgbm.



# Analisis de Modelo

## FEATURE IMPORTANCE

<img width="560" height="443" alt="image" src="https://github.com/user-attachments/assets/d9d2c6b6-fdf1-46e2-9a91-52328f8ddf36" />

Cabe mencionar que podemos ver cómo algunas de las features de nuestras hipotesis tienen alto nivel de relevancia para el modelo.


## Análisis de Shaps

<img width="169" height="176" alt="image" src="https://github.com/user-attachments/assets/5e7cbb99-4eb9-4002-8d6a-975f033f33d1" />

Un análisis de SHAP (SHapley Additive exPlanations) es una técnica para explicar cómo cada característica (variable) contribuye a la predicción de un modelo de machine learning. SHAP te dice cuánto y en qué dirección cada variable influye en la predicción de un modelo.

¿Para qué sirve?
•	Interpretar modelos complejos (como Random Forest, XGBoost, etc.)
•	Ver qué variables son más importantes globalmente.
•	Entender por qué un caso específico fue clasificado como positivo o negativo.
•	Detectar sesgos o problemas de confianza en modelos.


Realizaremos un análisis de shaps para entnder nuestro modelo y como las variables afectan su predicción. 
Es una análsis sencillo puedes ver la documentacion aqui:
https://shapash.readthedocs.io/en/latest/overview.html



y_pred_series = pd.Series(model.predict(X_test), index=X_test.index)
xpl = SmartExplainer(model=model)
xpl.compile(
    x=X_test,
    y_pred=y_pred_series
)


## Dashboard Generado
### Principales Features

<img width="595" height="452" alt="image" src="https://github.com/user-attachments/assets/391770e5-02fb-4113-95b2-80e660c9f273" />


### Total_Trans_Ct


<img width="720" height="478" alt="image" src="https://github.com/user-attachments/assets/7efd768d-64ae-404a-8780-7090e724d236" />


En la linea vertical podemos observar el valor shap, cuando es mayor a 0 tiene impacto positivo en la predicción del modelo, cuando es negativo tiene un impacto negativo en la predicción y cero neutro.

En este caso, cuando los valores de la variable Total_trans_ct, estan entre 0 y 60 aproximadamente, aporta positivamente a la predicción del modelo, en cambio para valores mayores a 60 aporta negativamente a la predicción.

En otras palabras si analizamos esta feature en particular, los clientes con Total_trans_ct, tienen mas probabilidades de desertar.

<img width="703" height="460" alt="image" src="https://github.com/user-attachments/assets/64143c73-8af0-4531-845f-3847b711e72c" />

<img width="710" height="459" alt="image" src="https://github.com/user-attachments/assets/14fed143-55de-4f06-9444-ab5a494f9866" />


# Conclusión

El objetivo principal de este proyecto fue desarrollar un modelo de clasificación capaz de predecir qué clientes de un banco tienen alta probabilidad de desertar (churn), con base en variables disponibles. 
Esta capacidad predictiva es crucial para permitirle al banco implementar estrategias preventivas de retención, con impacto directo en la reducción de pérdidas financieras.
Para la modelación, se emplearon dos algoritmos poderosos:
* Random Forest Classifier
* LightGBM (Light Gradient Boosting Machine)
Ambos modelos demostraron un alto poder predictivo y ambos clasificadores fueron capaces de separar correctamente las clases (clientes existentes vs. desertores).
Sin embargo LightGBM demostró tener mejores métricas generales.
 Entre las ventajas observadas:
* Mayor AUC (más capacidad para separar clases en diferentes umbrales).
* Menor MAE y RMSE (más precisión en las predicciones).
* Mejores tasas de recall para detectar desertores.
Por estas razones, LightGBM fue elegido como el modelo final, al ser más adecuado para este tipo de clasificación desequilibrada y por su rendimiento computacional eficiente.
Contar con un modelo predictivo de churn permite al banco:
* Identificar clientes en riesgo de abandonar con anticipación.
* Lanzar campañas personalizadas de retención (como beneficios, llamadas, ofertas).
* Optimizar recursos de marketing, enfocando esfuerzos en los perfiles con más probabilidad de desertar.
* Reducir la pérdida de ingresos por cancelaciones de cuentas o tarjetas.
Este proyecto demuestra el gran valor que aportan las herramientas de machine learning al negocio:
Machine Learning es herramienta fundamental para la competitividad, especialmente en industrias como la banca, donde la fidelización del cliente es esencial.

## Próximos pasos recomendados
* Integrar el modelo en procesos operativos del banco.
* Automatizar alertas para clientes en riesgo.
* Monitorear el rendimiento del modelo con nuevos datos en producción.
* Experimentar con estrategias de retención basadas en los factores más relevantes.

