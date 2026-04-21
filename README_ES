# Prediccion de Lesiones en Accidentes de Trafico — Madrid 2019–2023

Proyecto de clasificacion binaria supervisada para predecir si un accidente de trafico en Madrid resultara en al menos un herido, usando datos abiertos del Ayuntamiento de Madrid (2019–2023).

> [View this project in English](https://github.com/AlejandroBeldaFernandez/madrid-traffic-accidents/blob/main/README.md)

---

## Tabla de contenidos

1. [Definicion del problema](#definicion-del-problema)
2. [Valor de negocio](#valor-de-negocio)
3. [Dataset](#dataset)
4. [Retos y transformaciones de los datos](#retos-y-transformaciones-de-los-datos)
5. [Analisis exploratorio de datos](#analisis-exploratorio-de-datos)
6. [Metodologia](#metodologia)
7. [Pipeline de preprocesamiento](#pipeline-de-preprocesamiento)
8. [Modelos y optimizacion de hiperparametros](#modelos-y-optimizacion-de-hiperparametros)
9. [Resultados y evaluacion](#resultados-y-evaluacion)
10. [Explicabilidad — Analisis SHAP](#explicabilidad--analisis-shap)
11. [Problemas encontrados](#problemas-encontrados)
12. [Conclusiones](#conclusiones)
13. [Posibles mejoras](#posibles-mejoras)
14. [Requisitos](#requisitos)

---

## Definicion del problema

Los accidentes de trafico en Madrid generan miles de informes de incidentes cada año. Cada informe contiene informacion sobre la hora, ubicacion, tipo de accidente, vehiculos implicados, condiciones meteorologicas y el resultado para cada persona presente. Los equipos de respuesta de emergencia deben decidir rapidamente cuantos recursos y que tipo de personal enviar.

La pregunta que aborda este proyecto es:

> **Dada la informacion disponible inmediatamente despues de que se registra un accidente, podemos predecir si al menos una persona resultara herida?**

Se plantea como un problema de **clasificacion binaria**:

| Etiqueta | Significado |
|---|---|
| `injured` | Al menos una persona sufrio alguna lesion |
| `no_injury` | Ninguna persona implicada sufrio lesiones |

---

## Valor de negocio

Un predictor de lesiones fiable en tiempo real tiene valor operativo directo para los servicios de emergencia:

- **Despacho medico mas rapido.** Si el modelo marca un accidente como probable de resultar en lesiones, las ambulancias y el personal medico pueden enviarse simultaneamente con la policia, reduciendo el tiempo de respuesta.
- **Priorizacion de recursos.** En periodos de alta incidencia, el modelo puede ayudar a priorizar que accidentes requieren atencion medica inmediata y cuales pueden ser atendidos solo por agentes de trafico.
- **Apoyo a la decision, no sustituto.** El modelo esta disenado como herramienta de apoyo. Sus predicciones son probabilisticas y actuan como un primer filtro: los operadores humanos conservan la autoridad de decision final.

Las variables utilizadas (tipo de accidente, hora, distrito, numero de vehiculos, tipos de vehiculo, meteorologia) estan todas disponibles desde el informe inicial del incidente, lo que hace viable la inferencia en tiempo real.

---

## Dataset

- **Fuente:** Portal de datos abiertos del Ayuntamiento de Madrid
- **Periodo:** 2019–2023 (cinco años)
- **Granularidad:** Una fila por persona implicada en un accidente
- **Tamaño:** 212.511 filas antes de la limpieza
- **Algunas columnas:**

| Columna | Descripcion |
|---|---|
| `fecha` | Fecha del accidente |
| `hora` | Hora del accidente |
| `distrito` | Distrito de Madrid |
| `tipo_accidente` | Tipo de accidente (alcance, atropello, etc.) |
| `estado_meteorologico` | Condiciones meteorologicas |
| `tipo_vehiculo` | Tipo de vehiculo de la persona |
| `lesividad` | Gravedad de las lesiones de la persona |
| `coordenada_x_utm`, `coordenada_y_utm` | Coordenadas GPS (UTM) |

---

## Retos y transformaciones de los datos

### 1. Agregacion de nivel persona a nivel accidente

El dataset original esta a **nivel de persona**: una fila por persona por accidente. El objetivo de prediccion debe estar a **nivel de accidente**: una fila por accidente. Esto requirio un paso completo de agregacion:

- **Construccion del target:** Un accidente se etiqueta como `injured` si al menos una persona tiene una `lesividad` distinta a `"Sin asistencia sanitaria"`. Se calcula con una agregacion lambda que comprueba si alguna persona recibio atencion medica.
- **Flags de vehiculo:** Flags binarios creados para cada categoria de vehiculo (`flag_moto`, `flag_car`, `flag_van_truck`, `flag_bike_scooter`, `flag_bus`, `flag_other`) comprobando si alguna persona en el accidente conducia ese tipo de vehiculo.
- **Conteos:** `num_vehicles` (numero de vehiculos distintos) y `num_persons` (numero de personas implicadas) derivados por accidente.
- **Condiciones:** Distrito, tipo de accidente, meteorologia y franja horaria tomados del primer registro de cada accidente (consistentes dentro de un accidente).

### 2. Mapeo de gravedad

La columna original `lesividad` tiene ocho categorias. Se mapearon a una etiqueta binaria:

| Original | Etiqueta binaria |
|---|---|
| Sin asistencia sanitaria | `no_injury` |
| Todas las demas categorias (cualquier atencion medica) | `injured` |

### 3. Feature engineering

- **`time_slot`:** Hora extraida de `hora` y agrupada en: `dawn` (0–6), `morning` (7–11), `afternoon` (12–17), `rush_hour` (18–20), `night` (21–23).
- **`season`:** Mes mapeado a estacion meteorologica.
- **`year`:** Extraido de `fecha` para analisis temporal.

### 4. Limpieza de coordenadas

El dataset incluye coordenadas UTM para cada accidente, que se incluyeron como variables numericas en el modelo. Durante la fase de limpieza se descubrio un error sistematico de introduccion de datos: aproximadamente 11.000 accidentes tenian valores de coordenadas exactamente 1.000 veces mayores que el rango UTM valido de Madrid. Al representar los datos en un grafico el patron fue inmediatamente visible: los puntos anomalos formaban una replica perfecta a escala del trazado viario de la ciudad. La correccion consistio en dividir ambas columnas de coordenadas entre 1.000 en todas las filas afectadas, recuperando todos los registros sin perdida de datos.

---

## Analisis exploratorio de datos

El EDA se realizo a nivel de accidente tras la agregacion. Hallazgos principales:

### Distribucion del target

Tras la agregacion, la distribucion de clases cambia significativamente respecto a los datos a nivel de persona:

- **Nivel persona:** ~54% injured, ~46% no_injury
- **Nivel accidente:** ~84% injured, ~16% no_injury

Este cambio es esperado y correcto. Un accidente con cinco personas, cuatro sin lesiones y una con lesiones, sigue siendo un accidente con heridos. El dataset esta **desbalanceado**, siendo los accidentes sin heridos la clase minoritaria.

### Distrito

Todos los distritos muestran proporciones similares de accidentes con heridos. No hay ningun distrito que destaque dramaticamente, aunque existe cierta variacion.

### Franja horaria

La madrugada tiene la mayor proporcion de accidentes con heridos. Esto puede reflejar el mayor riesgo asociado a la fatiga y al menor volumen de trafico (menos accidentes en total pero proporcionalmente mas graves).

### Tipo de vehiculo

Las motocicletas, ciclomotores y autobuses son los tipos de vehiculo mas asociados a resultados con heridos. Las lesiones en moto y ciclomotor probablemente reflejan la falta de proteccion fisica del conductor; las lesiones en autobus pueden estar relacionadas con la masa y el tamaño del vehiculo.

### Condiciones meteorologicas

Las condiciones meteorologicas muestran una variacion minima en las tasas de lesiones. Las diferencias entre categorias son pequenas y el modelo trata la meteorologia como una señal debil.

### Tipo de accidente

Los atropellos y las salidas de via muestran la mayor proporcion de resultados con heridos. Los alcances y las colisiones laterales, a pesar de ser los tipos de accidente mas frecuentes, resultan en menos lesiones proporcionalmente.

### Numero de vehiculos

Los accidentes con dos o mas vehiculos tienen mas probabilidad de resultar en lesiones que los accidentes con un solo vehiculo.

### Alcohol y drogas

Las pruebas positivas a drogas muestran una asociacion ligeramente mayor con resultados con heridos que el alcohol. Sin embargo, ambos flags tienen poca importancia global debido a la rareza de los tests positivos.

### Tendencia temporal

La proporcion de accidentes con heridos alcanzo su punto maximo en 2020, probablemente influida por la reduccion del volumen de trafico durante los confinamientos por COVID-19: menos desplazamientos pero proporcionalmente de mayor riesgo. No se observa una tendencia clara al alza o a la baja a lo largo del periodo completo.

---

## Metodologia

El proyecto sigue un flujo de trabajo estructurado de aprendizaje supervisado:

1. **Carga y limpieza de datos** — gestion de valores nulos, correccion de tipos de datos, eliminacion de duplicados.
2. **Construccion del target** — agregacion de nivel persona a nivel accidente, derivacion de la etiqueta binaria.
3. **Feature engineering** — franjas horarias, estacion, flags de vehiculo, conteos.
4. **Analisis exploratorio de datos** — visualizacion de distribuciones y relaciones con el target.
5. **Division train/test** — division estratificada 80/20 para preservar la distribucion de clases.
6. **Pipeline de preprocesamiento** — codificacion y escalado mediante `ColumnTransformer`.
7. **Entrenamiento de modelos** — tres modelos entrenados con optimizacion de hiperparametros mediante Optuna.
8. **Evaluacion** — ROC AUC, balanced accuracy, F1 por clase, matrices de confusion.
9. **Explicabilidad** — valores SHAP sobre el mejor modelo (CatBoost).

---

## Pipeline de preprocesamiento

Un `ColumnTransformer` aplica diferentes transformaciones a cada grupo de variables:

| Grupo de variables | Transformacion |
|---|---|
| Categoricas de alta cardinalidad | Target Encoding |
| Categoricas ordinales | Ordinal Encoding |
| Categoricas de baja cardinalidad | One-Hot Encoding |
| Numericas | Standard Scaling |
| Flags binarios | Passthrough |
| Temporales | Passthrough |

Todos los transformadores se ajustan exclusivamente sobre el conjunto de entrenamiento. El Target Encoding usa suavizado con validacion cruzada para prevenir la fuga del target.

---

## Modelos y optimizacion de hiperparametros

Se entrenaron tres modelos, todos con gestion del desbalanceo de clases:

### Regresion Logistica

- **Gestion del desbalanceo:** `class_weight='balanced'`
- **Hiperparametros optimizados:** `C` (fuerza de regularizacion), `penalty` (`l1` / `l2`), solver seleccionado automaticamente segun la penalizacion
- **Optimizacion:** Optuna, 50 trials, CV estratificada de 5 folds, balanced accuracy como objetivo

### Random Forest

- **Gestion del desbalanceo:** `class_weight` (`balanced` / `balanced_subsample`)
- **Hiperparametros optimizados:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- **Optimizacion:** Optuna, 150 trials, CV estratificada de 5 folds, balanced accuracy como objetivo

### CatBoost

- **Gestion del desbalanceo:** `auto_class_weights` (`Balanced` / `SqrtBalanced`) como hiperparametro
- **Hiperparametros optimizados:** `depth`, `learning_rate`, `l2_leaf_reg`, `bagging_temperature`, `iterations`, `auto_class_weights`
- **Optimizacion:** Optuna con bucle manual `StratifiedKFold` (necesario para pasar `verbose=0` al fit de CatBoost) y 150 trials.

Se eligio balanced accuracy como metrica de validacion cruzada porque tiene en cuenta el desbalanceo de clases promediando el recall entre ambas clases, dando igual peso a la identificacion correcta de accidentes con y sin heridos.

---

## Resultados y evaluacion

### Rendimiento en el conjunto de test

| Modelo | ROC AUC | Balanced Accuracy | F1 no_injury | F1 injured | Macro F1 |
|---|---|---|---|---|---|
| Regresion Logistica | 0.851 | 0.787 | 0.55 | 0.85 | 0.70 |
| Random Forest | 0.862 | 0.790 | 0.58 | 0.88 | 0.73 |
| CatBoost | 0.873 | 0.801 | 0.59 | 0.88 | 0.73 |

### Observaciones clave

- Los tres modelos rinden a un nivel similar. Las diferencias en ROC AUC y balanced accuracy son pequenas (< 0.03).
- **CatBoost obtiene los mejores resultados** en todas las metricas.
- La precision en la clase `no_injury` es baja en todos los modelos (~0.40–0.45), lo que refleja la dificultad de identificar la clase minoritaria en este escenario desbalanceado.
- El recall en `injured` es alto (> 0.90 en todos los modelos), lo que significa que los modelos son eficaces detectando accidentes con heridos reales, que es el resultado operativamente mas importante.

### Recomendacion para produccion

En un entorno de produccion, **la Regresion Logistica seria el modelo preferido**. Ofrece una calidad comparable con un coste computacional mucho menor, es interpretable por perfiles no tecnicos y puede reentrenarse rapidamente a medida que llegan nuevos datos de accidentes. La ventaja de rendimiento de CatBoost es real pero marginal y no justifica la complejidad adicional en la mayoria de escenarios de despliegue.

---

## Explicabilidad — Analisis SHAP

Se aplicaron valores SHAP (SHapley Additive exPlanations) al modelo CatBoost para entender que impulsa las predicciones.

### Importancia global de variables

Las variables mas influyentes, ordenadas por valor SHAP absoluto medio:

1. **Tipo de accidente** — el predictor mas fuerte. Los atropellos y las salidas de via empujan fuertemente hacia `injured`; los alcances y los accidentes de aparcamiento empujan hacia `no_injury`.
2. **Numero de vehiculos** — las colisiones con multiples vehiculos estan fuertemente asociadas a lesiones.
3. **Distrito** — aporta una señal contextual moderada; algunos distritos son consistentemente mas peligrosos.
4. **Franja horaria** — la madrugada empuja ligeramente hacia `injured`.
5. **Flags de vehiculo** — la presencia de moto o ciclomotor aumenta la probabilidad de lesion.

**Variables menos relevantes:** La estacion y las condiciones meteorologicas muestran valores SHAP muy bajos, confirmando que estas variables no diferencian de forma significativa los accidentes con y sin heridos. Los flags de alcohol y drogas tambien son predictores debiles a nivel global debido a la rareza de los tests positivos.

### Analisis de errores

**Falsos negativos (injured predicho como no_injury):** Son accidentes donde la señal de lesion es debil, tipicamente de un solo vehiculo, tipo de accidente de bajo riesgo y fuera de las horas punta. El modelo no esta equivocado al ser incierto: las variables se parecen genuinamente a un accidente sin heridos.

**Falsos positivos (no_injury predicho como injured):** Estos accidentes comparten caracteristicas estructurales con la mayoria de accidentes con heridos: multiples vehiculos, hora de riesgo, tipo de accidente peligroso, pero no llegaron a causar lesiones. El modelo sobregenealiza los factores de riesgo estructurales, lo cual es un comportamiento razonable en un contexto de triaje.

Ambos tipos de error se concentran en las mismas variables principales, confirmando que el modelo falla en casos estructuralmente ambiguos y no por ruido aleatorio.

---

## Problemas encontrados

### 1. Baja precision en la clase minoritaria

El dataset esta muy desbalanceado (~84% injured, ~16% no_injury). A pesar de la ponderacion de clases, la precision para `no_injury` sigue siendo baja en todos los modelos. Esto significa que el modelo genera un numero significativo de falsos positivos: accidentes marcados como con heridos que no resultaron en lesiones. En un contexto operativo donde los falsos negativos (lesiones no detectadas) son mucho mas costosos que los falsos positivos, este es un compromiso aceptable, pero limita el uso del modelo como filtro binario estricto.

### 2. Dataset no preparado para el objetivo de prediccion

El dataset original fue recopilado con fines administrativos y estadisticos, no para machine learning. Registra resultados a nivel de persona a posteriori, lo que obliga a reconstruir la informacion a nivel de accidente mediante agregacion. Decisiones como gestionar registros de gravedad contradictorios dentro del mismo accidente y derivar una etiqueta unica coherente requirieron un diseño cuidadoso. La variable target no existe en los datos brutos: tuvo que construirse completamente.

### 3. Valores de coordenadas corruptos

El dataset incluye coordenadas GPS UTM para cada accidente. Durante la limpieza descubrimos que ~11.000 registros tenian valores de coordenadas exactamente 1.000 veces mayores que el rango UTM valido de Madrid, un error sistematico de introduccion de datos. El problema no era obvio inspeccionando los numeros en bruto, pero quedo claro al representarlos graficamente: los puntos anomalos formaban una replica perfecta a escala del trazado viario de Madrid. La correccion fue directa: dividir ambas columnas de coordenadas entre 1.000 en todas las filas afectadas, sin perder ningun registro. Las coordenadas corregidas se incluyeron como variables numericas en el modelo.

---

## Conclusiones

Este proyecto demuestra que es posible predecir lesiones en accidentes de trafico en Madrid usando unicamente informacion disponible en el momento en que se registra el incidente, antes de que se realice ninguna evaluacion medica.

El mejor modelo (CatBoost) alcanza un ROC AUC de 0.873 y una balanced accuracy de 0.801. Mas importante aun, los dos predictores mas fuertes son el tipo de accidente y el numero de vehiculos implicados, ambos capturados en el informe inicial del agente interviniente. Esto significa que la prediccion puede realizarse en tiempo real, en el lugar del accidente, sin necesidad de recopilar datos adicionales.

La implicacion practica es concreta: los servicios de coordinacion de emergencias podrian usar un modelo de este tipo para priorizar el despacho de recursos. Los accidentes marcados como de alta probabilidad de lesion activarian una asignacion mas rapida de ambulancias o alertarian a unidades medicas cercanas, reduciendo el tiempo de respuesta en los casos donde mas importa.

La principal limitacion actual es la precision en la clase sin heridos, que genera falsos positivos. En un contexto de emergencias este es el tipo de error preferible: es mas seguro enviar recursos innecesariamente que no responder a una lesion real. Sin embargo, tiene implicaciones de coste operativo que deberian evaluarse frente a las restricciones presupuestarias de la organizacion que lo desplegara.

Para un despliegue en produccion, la Regresion Logistica es el modelo recomendado. Su rendimiento es suficientemente proximo al de CatBoost como para que su interpretabilidad y bajo coste computacional justifiquen la eleccion, especialmente en entornos donde las decisiones deben ser auditables y explicables para perfiles no tecnicos.

---

## Posibles mejoras

- **Remuestreo con SMOTE o ADASYN.** Aplicar sobremuestreo sintetico de la clase minoritaria (SMOTE, BorderlineSMOTE, ADASYN) o estrategias combinadas de sobre y submuestreo de la libreria `imbalanced-learn` podria mejorar el recall y la precision en la clase `no_injury`.
- **Clasificadores nativos de imbalanced-learn.** Metodos ensemble disenados especificamente para problemas desbalanceados como `BalancedRandomForest`, `EasyEnsembleClassifier` o `RUSBoostClassifier` podrian superar a los modelos actuales, especialmente en la clase minoritaria.
- **Optimizacion del umbral de clasificacion.** En lugar de usar el umbral de clasificacion por defecto de 0.5, ajustarlo en un conjunto de validacion para minimizar los falsos negativos sujeto a una restriccion de precision alinearia mejor el modelo con los requisitos operativos.

---

## Requisitos

pandas
numpy
matplotlib
seaborn
scikit-learn
catboost
optuna
shap

Instalar todas las dependencias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost optuna shap imbalanced-learn
```

---

*Fuente de datos: https://www.kaggle.com/datasets/leomed666/traffic-accidents-in-madrid-spain-from-2019-2023*
