# API de Clasificación de Cáncer de Mama

Este proyecto implementa un servicio web FastAPI para la clasificación de cáncer de mama utilizando el conjunto de datos Wisconsin Breast Cancer de scikit-learn. La API permite a los usuarios hacer predicciones a través de una interfaz REST.

## Características

- API REST construida con FastAPI
- Clasificación de cáncer de mama usando el algoritmo Random Forest
- Entrenamiento automático del modelo en la primera ejecución
- Registro de actividades completo
- Documentación Swagger
- Guía de integración con Google Colab

## Requisitos

- Python 3.8+
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clona este repositorio
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecutando la API

Inicia el servidor API con:

```bash
python main.py
```

O usando uvicorn directamente:

```bash
uvicorn main:app --reload
```

La API estará disponible en http://localhost:8000

## Documentación de la API

Una vez que el servidor esté en funcionamiento, puedes acceder a la documentación interactiva de la API en:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints de la API

- `GET /`: Mensaje de bienvenida
- `GET /health`: Endpoint de verificación de salud
- `GET /features`: Obtener la lista de nombres de características requeridas para la predicción
- `POST /predict`: Realizar una predicción de clasificación de cáncer de mama

## Realizando Predicciones

Para hacer una predicción, envía una solicitud POST al endpoint `/predict` con una carga JSON que contenga el arreglo de características:

```json
{
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
}
```

La respuesta incluirá la predicción (0 para maligno, 1 para benigno), la etiqueta de predicción y la probabilidad:

```json
{
  "prediction": 0,
  "prediction_label": "malignant",
  "probability": 0.98
}
```

## Pruebas con Curl

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
}'
```

## Registro de Actividades

La API incluye un registro completo de actividades tanto en la consola como en un archivo (`api.log`). Todas las solicitudes a la API, el entrenamiento del modelo y las actividades de predicción se registran para monitoreo y depuración.

## Integración con Google Colab

Este proyecto incluye un notebook de Google Colab (`breast_cancer_api_colab.ipynb`) que demuestra cómo desplegar y usar la API en un entorno de Google Colab. El notebook cubre:

- Configuración del entorno en Google Colab
- Comprensión de la estructura del código
- Ejecución del servidor FastAPI con ngrok para acceso público
- Realización de predicciones usando la API
- Exploración de la documentación de la API
- Monitoreo y registro de actividades

Para usar el notebook de Colab:

1. Sube el archivo `breast_cancer_api_colab.ipynb` a Google Colab
2. Sigue las instrucciones paso a paso en el notebook
