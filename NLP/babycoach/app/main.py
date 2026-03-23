from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api.chat import router as chat_router
from .api.recommend import router as recommend_router
from .graph import get_compiled_graph
from .ui.app_ui import get_ui_html


def create_app() -> FastAPI:
    app = FastAPI(title="BabyCoach PoC")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static assets (icons/images). Path is fixed and used in UI only.
    assets_dir = os.path.join(os.path.dirname(__file__), "ui", "assets")
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    app.include_router(recommend_router)
    app.include_router(chat_router)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return get_ui_html()

    @app.get("/mom", response_class=HTMLResponse)
    def mom() -> str:
        html_path = os.path.join(os.path.dirname(__file__), "ui", "babycoach_mom.html")
        with open(html_path, encoding="utf-8") as f:
            return f.read()

    @app.get("/mom2", response_class=HTMLResponse)
    def mom2() -> str:
        html_path = os.path.join(os.path.dirname(__file__), "ui", "babycoach_mom2.html")
        with open(html_path, encoding="utf-8") as f:
            return f.read()

    @app.get("/mom3", response_class=HTMLResponse)
    def mom3() -> str:
        html_path = os.path.join(os.path.dirname(__file__), "ui", "babycoach_mom3.html")
        with open(html_path, encoding="utf-8") as f:
            return f.read()

    @app.get("/mom4", response_class=HTMLResponse)
    def mom4() -> str:
        html_path = os.path.join(os.path.dirname(__file__), "ui", "babycoach_mom4.html")
        with open(html_path, encoding="utf-8") as f:
            return f.read()

    @app.get("/mom5", response_class=HTMLResponse)
    def mom5() -> str:
        html_path = os.path.join(os.path.dirname(__file__), "ui", "babycoach_mom5.html")
        with open(html_path, encoding="utf-8") as f:
            return f.read()

    @app.get("/doctor", response_class=HTMLResponse)
    def doctor() -> str:
        html_path = os.path.join(os.path.dirname(__file__), "ui", "babycoach_doctor.html")
        with open(html_path, encoding="utf-8") as f:
            return f.read()

    @app.get("/health")
    def health() -> dict:
        # Ensure graph can compile at runtime (smoke check).
        get_compiled_graph()
        return {"status": "ok"}

    return app


app = create_app()

