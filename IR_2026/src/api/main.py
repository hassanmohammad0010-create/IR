#  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.common.db import close_pool

from src.api.routes.search_routes import router as search_router
from src.api.routes.evaluation_routes import router as evaluation_router
from src.api.routes.websocket_routes import router as websocket_router
from src.api.routes.rag_routes import router as rag_router
from src.api.routes.clustering_routes import router as clustering_router
from src.api.routes.ltr_routes import router as ltr_router

from src.websocket.websocket_logger import configure_logging

configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting IR System API ...")
    import src.api.dependencies.container

    logger.info("IR System API is ready.")
    yield
    close_pool()
    logger.info("IR System API shutting down.")


app = FastAPI(
    title="IR System API",
    description=(
        "Information Retrieval system supporting BM25, TF-IDF, "
        "Embedding, Hybrid, RAG (online, OpenRouter), LTR (offline, "
        "Logistic Regression), and Clustering — each selectable "
        "independently from the UI."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)
app.include_router(evaluation_router)
app.include_router(websocket_router)
app.include_router(rag_router)
app.include_router(clustering_router)
app.include_router(ltr_router)
