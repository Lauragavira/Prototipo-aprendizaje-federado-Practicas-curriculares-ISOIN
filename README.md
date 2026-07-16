# Simulador de aprendizaje federado con Flower

Prototipo desarrollado para configurar, ejecutar y analizar entrenamientos de **aprendizaje federado** desde una interfaz web. El proyecto combina:

- **Streamlit** para el panel de control y las visualizaciones.
- **Flower** para coordinar el servidor, los SuperNodes y las rondas federadas.
- **PyTorch** para definir y entrenar los modelos locales.
- **Plotly** para comparar métricas globales y resultados por distribuidor.

El caso de uso logístico predice directamente los **minutos de retraso** (`retraso_min`) de cada distribuidor. Los datos permanecen asociados a cada cliente y el servidor agrega los parámetros de los modelos locales, no los registros originales.

> El archivo conserva el nombre `task_regresion_logistica.py` por compatibilidad con la interfaz, pero el modelo implementado es una **regresión lineal de retrasos logísticos**, no una regresión logística de clasificación.

## Funcionalidades principales

- Inicio y parada del **SuperLink** desde Streamlit.
- Detección de la IP local del ordenador servidor.
- Ejecución con varios SuperNodes, en un único equipo o en equipos diferentes.
- Selección dinámica de tareas disponibles en `task/`.
- Selección de entre **dos y cuatro distribuidores** para compararlos.
- Configuración de rondas, participación de clientes, estrategia, `learning rate` y `batch size`.
- Estrategias federadas `FedAvg`, `FedProx` y `FedMedian`.
- Ruido gaussiano experimental sobre los parámetros locales.
- Checkpoints por ronda y reanudación del entrenamiento.
- Métricas globales y métricas individuales por distribuidor.
- Comparativas de MAE, MSE, RMSE, R², retraso medio real y predicho, diferencia absoluta y sesgo.
- Gráficas de evolución por rondas y tabla final de resultados.

### Componentes

| Archivo | Función |
|---|---|
| `app.py` | Panel Streamlit, configuración, procesos y gráficas. |
| `server.py` | Estrategia Flower, agregación, métricas y checkpoints. |
| `client.py` | Carga dinámica de tareas, asignación de distribuidores, entrenamiento y evaluación local. |
| `task/task_mnist.py` | Clasificación MNIST con distribución IID. |
| `task/task_noIID_mnist.py` | Clasificación MNIST con distribución No-IID. |
| `task/task_regresion_logistica.py` | Regresión de `retraso_min`. |
| `distribuidores/` | Cuatro CSV logísticos de ejemplo. |
| `pyproject.toml` | Dependencias y componentes de la Flower App. |

## Estructura del repositorio

```text
.
├── app.py
├── client.py
├── server.py
├── pyproject.toml
├── README.md
├── distribuidores/
│   ├── distribuidor_1.csv
│   ├── distribuidor_2.csv
│   ├── distribuidor_3.csv
│   └── distribuidor_4.csv
├── task/
│   ├── __init__.py
│   ├── task_mnist.py
│   ├── task_noIID_mnist.py
│   └── task_regresion_logistica.py
└── checkpoint/
```

## Regresión de retrasos logísticos

### Variable objetivo

El modelo predice:

```text
retraso_min
```

La salida es un número real expresado en minutos. 

### Variables de entrada

1. `hora_salida_min`
2. `distancia_km`
3. `numero_entregas`
4. `peso_total_kg`
5. `duracion_estimada_min`
6. `hora_llegada_estimada_min`

### Preparación de los datos

- Las columnas se convierten a valores numéricos.
- Se eliminan filas con datos inválidos en las variables necesarias.
- Cada CSV contiene 150 registros.
- La división es del 80 % para entrenamiento y 20 % para prueba: 120 y 30 registros por distribuidor.
- Se utiliza `random_state=42` para mantener la misma división entre ejecuciones.
- Todos los clientes aplican un **escalado fijo compartido**. No se ajusta un `StandardScaler` independiente en cada cliente, porque eso haría que unos mismos pesos globales representasen escalas distintas.

### Modelo y entrenamiento local

- Capa lineal de seis entradas y una salida.
- Función de pérdida: `MSELoss`.
- Optimizador: SGD con `momentum=0.9`.
- Cinco épocas locales por ronda.
- Recorte de gradiente con norma máxima de 10.

## Asignación de distribuidores a clientes

La selección realizada en Streamlit se guarda en `selected_distributors`. El `partition-id` de cada SuperNode indica la posición que ocupa en esa lista.

Ejemplo:

```text
Distribuidores seleccionados: [2, 3, 4]
partition-id=0 → distribuidor 2
partition-id=1 → distribuidor 3
partition-id=2 → distribuidor 4
```

El número de SuperNodes conectados debe coincidir con el número de distribuidores seleccionados y `num-partitions` debe tener ese mismo valor.

## Métricas

| Métrica | Interpretación |
|---|---|
| `Loss` | Pérdida global de evaluación. En la regresión corresponde al error cuadrático medio. Menor es mejor. |
| `MAE` | Media del error absoluto, expresada en minutos. Menor es mejor. |
| `MSE` | Media de los errores al cuadrado, expresada en minutos². Penaliza especialmente los errores grandes. |
| `RMSE` | Raíz del MSE, expresada en minutos. Menor es mejor. |
| `R²` | Proporción de variabilidad explicada. Un valor cercano a 1 es mejor; un valor negativo indica un rendimiento inferior a predecir siempre la media. |
| Retraso real medio | Media de `retraso_min` en el conjunto de prueba del distribuidor. |
| Retraso predicho medio | Media de las predicciones del modelo global. |
| Diferencia media | Valor absoluto entre el retraso real medio y el predicho medio. No es lo mismo que el MAE. |
| Sesgo medio | `predicho − real`. Positivo: sobreestimación. Negativo: infraestimación. |

El servidor calcula las métricas globales mediante una media ponderada por el número de ejemplos de evaluación de cada cliente. También conserva los resultados individuales en `metrics.json`.

## Visualizaciones del panel

### Métricas globales

- Tarjetas con el último valor y la variación respecto a la ronda anterior.
- Curvas por ronda para las métricas seleccionadas.

### Comparación por distribuidor

La aplicación crea cuatro pestañas:

1. **Comparación de errores**
   - MAE y RMSE agrupados.
   - R² por distribuidor.
   - Diferencia `RMSE − MAE`, útil para detectar errores extremos.

2. **Retraso real y predicho**
   - Retraso real medio frente al predicho.
   - Diferencia absoluta entre las medias.
   - Sesgo medio.
   - Diagrama de dispersión respecto a la diagonal ideal.

3. **Evolución por rondas**
   - MAE, RMSE, R² y diferencia media de cada distribuidor durante el entrenamiento.

4. **Tabla completa**
   - Resumen numérico de todas las métricas de la última ronda.

## Requisitos

- Python 3.11 o superior.
- Flower 1.32.x recomendado para reproducir el entorno utilizado.
- PyTorch y Torchvision.
- Streamlit.
- NumPy, Pandas, Plotly y scikit-learn.

## Instalación

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Comprobación:

```powershell
python --version
flwr --version
python -m pip -V
```

### Linux o macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Configuración de Flower

La aplicación inicia el entrenamiento con:

```bash
flwr run . red-fisica --stream
```

Por ello debe existir una conexión llamada `red-fisica` en el archivo global de Flower.

### 1. Localizar el archivo

```powershell
flwr config list
```

El comando muestra una ruta similar a:

```text
C:\Users\USUARIO\.flwr\config.toml
```

### 2. Añadir la conexión

Añade al final de `config.toml`:

```toml
[superlink.red-fisica]
address = "127.0.0.1:9093"
insecure = true
```

No debe guardarse como `config.toml.txt` ni colocarse junto a `pyproject.toml`.

### 3. Verificar

```powershell
flwr config list
```

La lista debe incluir `red-fisica`.

## Ejecución

### 1. Abrir Streamlit

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

### 2. Levantar el SuperLink

Pulsa **Levantar SuperLink** en la barra lateral.

- `9092`: conexión de los SuperNodes.
- `9093`: Control API utilizada por `flwr run`.
- `9091`: comunicación interna con la ServerApp.

### 3. Conectar los SuperNodes

#### Tres clientes en el mismo ordenador

Abre tres terminales independientes (si quieres hacerlo con 3 distribuidores) y activa el entorno en cada una. Cada SuperNode necesita un `partition-id` y un puerto `clientappio` diferentes.

```powershell
# Cliente 1
flower-supernode --insecure `
  --superlink 127.0.0.1:9092 `
  --clientappio-api-address 127.0.0.1:9094 `
  --node-config "partition-id=0 num-partitions=3"
```

```powershell
# Cliente 2
flower-supernode --insecure `
  --superlink 127.0.0.1:9092 `
  --clientappio-api-address 127.0.0.1:9095 `
  --node-config "partition-id=1 num-partitions=3"
```

```powershell
# Cliente 3
flower-supernode --insecure `
  --superlink 127.0.0.1:9092 `
  --clientappio-api-address 127.0.0.1:9096 `
  --node-config "partition-id=2 num-partitions=3"
```

En algunas instalaciones el ejecutable se llama `flwr-supernode`; puede sustituirse `flower-supernode` por ese nombre.

#### Clientes en ordenadores diferentes

En cada equipo cliente:

```powershell
flower-supernode --insecure `
  --superlink IP_DEL_SERVIDOR:9092 `
  --clientappio-api-address 127.0.0.1:9094 `
  --node-config "partition-id=N num-partitions=TOTAL"
```

- `IP_DEL_SERVIDOR`: IP mostrada por Streamlit.
- `N`: identificador único desde 0 hasta `TOTAL - 1`.
- `TOTAL`: número de distribuidores seleccionados.

Si hay un único SuperNode por equipo, todos pueden usar localmente el puerto `9094`, porque no comparten máquina.

### 4. Iniciar el entrenamiento

En Streamlit:

1. Selecciona `task_regresion_logistica`.
2. Elige entre dos y cuatro distribuidores.
3. Marca MAE, MSE, RMSE y R² según el análisis deseado.
4. Configura rondas, estrategia, participación, `learning rate` y `batch size`.
5. Pulsa **Iniciar ServerApp (Entrenamiento)**.

Cuando aparece:

```text
[INIT]
Requesting initial parameters from one random client
```

el servidor está funcionando, pero todavía espera al menos un SuperNode conectado. No es un error.

## Archivos generados

| Elemento | Contenido |
|---|---|
| `run_config.json` | Configuración elegida en Streamlit. |
| `metrics.json` | Métricas globales y resultados por distribuidor en cada ronda. |
| `checkpoint/<tarea>/checkpoint_round_N.npz` | Parámetros del modelo global. |
| `stop_training.txt` | Señal temporal de parada. |

## Checkpoints

- Sin la opción **Usar checkpoints**, la app elimina los checkpoints anteriores al iniciar.
- Con la opción activada, el servidor carga el checkpoint con el número de ronda más alto.
- Los checkpoints de la antigua clasificación binaria no son compatibles con el modelo de una salida. Deben eliminarse antes de usar la regresión actual.

## Privacidad y alcance del prototipo

La aplicación asocia varios valores de `ε` con niveles predefinidos de ruido gaussiano y suma ese ruido a los parámetros locales. Esta función es una simulación experimental y **no constituye privacidad diferencial formal**, porque no incorpora recorte de actualizaciones, cálculo de sensibilidad ni contador de privacidad.

Además, `--insecure` desactiva TLS. Solo debe utilizarse en una red local controlada.

En el prototipo, los cuatro CSV están dentro del repositorio para facilitar las pruebas. En un despliegue real, cada SuperNode debería tener acceso únicamente a los datos de su propio distribuidor.

## Problemas frecuentes

### `SuperLink connection 'red-fisica' not found`

`red-fisica` no existe en el archivo mostrado por `flwr config list`. Añade la sección correspondiente a `~/.flwr/config.toml`.

### El servidor se queda en `Requesting initial parameters`

No hay suficientes SuperNodes conectados. Inicia tantos clientes como distribuidores seleccionados.

### Puerto ocupado al iniciar varios clientes

Asigna un valor diferente a `--clientappio-api-address` para cada SuperNode que se ejecute en el mismo equipo.

### Todos los clientes usan el mismo distribuidor

Comprueba que cada terminal tiene un `partition-id` diferente y que `num-partitions` coincide con el número de distribuidores seleccionados.

### Error al cargar un checkpoint

Elimina los archivos antiguos de:

```text
checkpoint/task_regresion_logistica/
```

## Limitaciones y mejoras futuras

- El modelo es lineal y puede no capturar relaciones complejas.
- Las variables categóricas como tráfico, clima o zona todavía no se utilizan.
- El escalado compartido utiliza constantes definidas manualmente.
- Los datos de ejemplo son pequeños y deben validarse con datos reales antes de extraer conclusiones empresariales.
- La privacidad diferencial es solo aproximada.
- La ejecución de laboratorio utiliza comunicaciones sin cifrar.
- La interfaz muestra un comando base de SuperNode, pero los identificadores de partición deben añadirse manualmente.

## Referencias

- Flower Framework: <https://flower.ai/docs/framework/>
- Flower Deployment Runtime: <https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html>
- Flower Architecture: <https://flower.ai/docs/framework/explanation-flower-architecture.html>
- Flower CLI: <https://flower.ai/docs/framework/ref-api-cli.html>
- Streamlit: <https://docs.streamlit.io/>
- PyTorch: <https://pytorch.org/docs/stable/>
- scikit-learn: <https://scikit-learn.org/stable/>

## Autores

Proyecto desarrollado durante las prácticas curriculares en ISOIN por los alumnos Laura Gavira y Juan Vázquez, de la Universidad de Sevilla.
