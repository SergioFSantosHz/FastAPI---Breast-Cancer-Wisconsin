from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import uvicorn
import pickle
import numpy as np
import logging
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Clasificación de Cáncer de Mama",
              description="API para clasificación de cáncer de mama usando el conjunto de datos Wisconsin Breast Cancer",
              version="1.0.0")

# Definir modelo de solicitud
class EntradaCaracteristicas(BaseModel):
    caracteristicas: list
    
    class Config:
        schema_extra = {
            "example": {
                "caracteristicas": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
            }
        }

# Definir modelo de respuesta
class RespuestaPrediccion(BaseModel):
    prediccion: int
    etiqueta_prediccion: str
    probabilidad: float

# Ruta del modelo
RUTA_MODELO = "modelo_cancer_mama.pkl"

# Función para entrenar y guardar el modelo si no existe
def entrenar_y_guardar_modelo():
    logger.info("Cargando conjunto de datos de cáncer de mama")
    datos = load_breast_cancer()
    X = datos.data
    y = datos.target
    nombres_caracteristicas = datos.feature_names
    nombres_objetivos = datos.target_names
    
    logger.info(f"Conjunto de datos cargado con {X.shape[0]} muestras y {X.shape[1]} características")
    
    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar un modelo Random Forest
    logger.info("Entrenando clasificador Random Forest")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    logger.info(f"Modelo entrenado con precisión: {precision:.4f}")
    
    # Guardar el modelo, nombres de características y nombres de objetivos
    datos_modelo = {
        "modelo": modelo,
        "nombres_caracteristicas": nombres_caracteristicas,
        "nombres_objetivos": nombres_objetivos
    }
    
    with open(RUTA_MODELO, 'wb') as f:
        pickle.dump(datos_modelo, f)
    
    logger.info(f"Modelo guardado en {RUTA_MODELO}")
    return datos_modelo

# Función para cargar el modelo
def obtener_modelo():
    if not os.path.exists(RUTA_MODELO):
        logger.info(f"Archivo de modelo {RUTA_MODELO} no encontrado. Entrenando un nuevo modelo.")
        return entrenar_y_guardar_modelo()
    
    try:
        logger.info(f"Cargando modelo desde {RUTA_MODELO}")
        with open(RUTA_MODELO, 'rb') as f:
            datos_modelo = pickle.load(f)
        logger.info("Modelo cargado exitosamente")
        return datos_modelo
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        return entrenar_y_guardar_modelo()

# Evento de inicio para asegurar que el modelo se carga cuando la aplicación inicia
@app.on_event("startup")
async def evento_inicio():
    logger.info("Iniciando el servidor API")
    obtener_modelo()
    logger.info("Servidor API iniciado exitosamente")

# Endpoint raíz
@app.get("/")
def leer_raiz():
    logger.info("Endpoint raíz accedido")
    return {"mensaje": "Bienvenido a la API de Clasificación de cáncer de mama"}

# Endpoint de verificación de salud
@app.get("/salud")
def verificacion_salud():
    logger.info("Endpoint de verificación de salud accedido")
    return {"estado": "saludable"}

# Endpoint de predicción
@app.post("/predecir", response_model=RespuestaPrediccion)
def predecir(datos_entrada: EntradaCaracteristicas, datos_modelo=Depends(obtener_modelo)):
    try:
        logger.info("Endpoint de predicción accedido")
        
        # Extraer componentes del modelo
        modelo = datos_modelo["modelo"]
        nombres_caracteristicas = datos_modelo["nombres_caracteristicas"]
        nombres_objetivos = datos_modelo["nombres_objetivos"]
        
        # Validar características de entrada
        if len(datos_entrada.caracteristicas) != len(nombres_caracteristicas):
            mensaje_error = f"Se esperaban {len(nombres_caracteristicas)} características, pero se recibieron {len(datos_entrada.caracteristicas)}"
            logger.error(mensaje_error)
            raise HTTPException(status_code=400, detail=mensaje_error)
        
        # Hacer predicción
        caracteristicas = np.array(datos_entrada.caracteristicas).reshape(1, -1)
        prediccion = modelo.predict(caracteristicas)[0]
        probabilidad = modelo.predict_proba(caracteristicas)[0][prediccion]
        
        # Crear respuesta
        respuesta = {
            "prediccion": int(prediccion),
            "etiqueta_prediccion": nombres_objetivos[prediccion],
            "probabilidad": float(probabilidad)
        }
        
        logger.info(f"Predicción realizada: {respuesta}")
        return respuesta
    
    except Exception as e:
        mensaje_error = f"Error al realizar la predicción: {str(e)}"
        logger.error(mensaje_error)
        raise HTTPException(status_code=500, detail=mensaje_error)

# Endpoint para obtener nombres de características
@app.get("/caracteristicas")
def obtener_caracteristicas(datos_modelo=Depends(obtener_modelo)):
    logger.info("Endpoint de características accedido")
    return {"nombres_caracteristicas": datos_modelo["nombres_caracteristicas"].tolist()}

# Ejecutar la aplicación
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)