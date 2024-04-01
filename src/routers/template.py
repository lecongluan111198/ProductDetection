from fastapi import APIRouter, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi import Depends, FastAPI, Request, Response
from ..utils import json_utils, profiler

router = APIRouter()
templates = Jinja2Templates(directory="resources/static")

@ router.get("/profilers", response_class=HTMLResponse)
def getProfilers(request: Request, refresh: int = 2):
    thread_stats = []
    func_stats = []
    for k, v in profiler.thread_stats.copy().items():
        thread_stats.append(v.__dict__)
    for k, v in profiler.func_stats.copy().items():
        func_stats.append(v.__dict__)
    return templates.TemplateResponse('html/profiler.html', {
        'request': request,
        "thread_stats": thread_stats,
        "func_stats": func_stats,
        "refresh": refresh
    })