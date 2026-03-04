from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


def register_exception_handlers(app: FastAPI) -> None:
    """
    注册全局异常拦截器。
    """

    @app.exception_handler(HTTPException)
    async def http_exception_handler(  # type: ignore[unused-ignore]
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        payload: Dict[str, Any] = {
            "code": exc.status_code,
            "message": exc.detail,
        }
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(  # type: ignore[unused-ignore]
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        payload: Dict[str, Any] = {
            "code": 500,
            "message": "Internal Server Error",
        }
        return JSONResponse(status_code=500, content=payload)
