from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import explain, health, history, models, predict
from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Cow vs Sheep Classifier API", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["health"])
    app.include_router(predict.router, tags=["prediction"])
    app.include_router(explain.router, tags=["prediction"])
    app.include_router(models.router, tags=["models"])
    app.include_router(history.router, tags=["history"])
    return app


app = create_app()
