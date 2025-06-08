# Servicio de Predicciones

Este microservicio está diseñado para proporcionar predicciones de riesgo académico para estudiantes utilizando modelos de machine learning. El servicio analiza diferentes aspectos del rendimiento estudiantil y genera predicciones de riesgo en tiempo real.

## Características Principales

- Predicción de riesgo académico basada en múltiples enfoques:
  - Rendimiento académico
  - Participación
  - Entrega de tareas
  - Interacción con mentores
- API RESTful con FastAPI
- Autenticación mediante tokens OAuth2
- Almacenamiento de predicciones en Supabase
- Procesamiento asíncrono de predicciones
- Documentación automática de API con OpenAPI/Swagger

## Tecnologías Utilizadas

- **Backend Framework**: FastAPI 0.109.2
- **Base de Datos**: PostgreSQL (Supabase)
- **ORM**: SQLAlchemy 2.0.27
- **Machine Learning**: 
  - LightGBM 4.3.0
  - scikit-learn 1.4.2
  - pandas 2.2.1
  - numpy 1.26.4
- **Autenticación**: python-jose[cryptography] 3.3.0
- **Variables de Entorno**: python-dotenv 1.0.1

## Requisitos Previos

- Python 3.8 o superior
- PostgreSQL (o acceso a Supabase)
- pip (gestor de paquetes de Python)

## Instalación Local

1. Clonar el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
cd service-predictions
```

2. Crear un entorno virtual e instalar dependencias:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configurar variables de entorno:
   - Copiar el archivo `envs/.env.example` a `.env` en la raíz del proyecto
   - Completar las variables de entorno necesarias:
     - `PROJECT_NAME`: Nombre del proyecto (por defecto: "Prediction Service")
     - `VERSION`: Versión del proyecto (por defecto: "1.0.0")
     - `SUPABASE_DB_URL`: URL de conexión a Supabase
     - `SUPABASE_PASSWORD`: Contraseña de Supabase
     - `API_AUTH_URL`: URL del servicio de autenticación

   Para producción, se puede usar el archivo `envs/.env.prod` como base.

4. Ejecutar el servidor de desarrollo:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 4001
```

El servidor estará disponible en `http://localhost:4001`

## Instalación con Docker

1. Construir la imagen:
```bash
docker build -t service-predictions .
```

2. Ejecutar el contenedor:
```bash
docker run -d -p 4001:4001 --env-file envs/.env.prod --name service-predictions service-predictions
```

El servicio estará disponible en `http://localhost:4001`

## Uso de la API

El servicio expone los siguientes endpoints principales:

- `POST /internal/process/predictions/process_student_data`: Procesa datos de un estudiante y genera predicciones
- `POST /internal/process/predictions/train_models`: Entrena los modelos de predicción
- `GET /internal/process/predictions/get_last_prediction`: Obtiene la última predicción para un estudiante
- `GET /api/v1/health`: Verifica el estado del servicio

La documentación completa de la API está disponible en `/docs` cuando el servidor está en ejecución.

## Estructura del Proyecto

```
service-predictions/
├── app/
│   ├── controller/         # Controladores de la API
│   ├── core/              # Configuración central
│   ├── models/            # Modelos de base de datos
│   ├── schemas/           # Esquemas Pydantic
│   ├── services/          # Lógica de negocio
│   └── utils/             # Utilidades
├── main.py               # Punto de entrada de la aplicación
├── requirements.txt      # Dependencias del proyecto
└── .env                 # Variables de entorno
```

## Desarrollo

Para ejecutar el proyecto en modo desarrollo con VS Code:

1. Abrir el proyecto en VS Code
2. Seleccionar la configuración "Debug FastAPI (Uvicorn)" en el panel de depuración
3. Iniciar la depuración (F5)

## Licencia

© 2025 Lucio Gabriel Abaca. Todos los derechos reservados. 