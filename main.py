from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.controller.router import api_router
from fastapi.openapi.utils import get_openapi
import requests
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.utils.logging_config import setup_logging

setup_logging()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "OAuth2PasswordBearer": {
            "type": "oauth2",
            "flows": {
                "password": {
                    "tokenUrl": f"{settings.API_AUTH_URL}/login",
                    "scopes": {}
                }
            }
        }
    }
    openapi_schema["security"] = [{"OAuth2PasswordBearer": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir los routers de la API
app.include_router(api_router)

@app.middleware("http")
async def verify_access_token(request: Request, call_next):
    if (request.url.path.split('/')[1] or False) == "internal":
        return await call_next(request)
    
    if request.url.path in ["/docs", "/openapi.json", "/internal", '/api/v1/health']: 
        return await call_next(request)
    
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return JSONResponse(status_code=401, content={"detail": "Access token not found or invalid"})

    access_token = auth_header.split(" ")[1]

    response = requests.get(
        f"{settings.API_AUTH_URL}/me",
        headers={
            "Authorization": f"Bearer {access_token}",
        }
    )
    if response.status_code != 200:
        return JSONResponse(status_code=401, content={"detail": "Access token invalid"})
    
    request.state.user = response.json()
    return await call_next(request)