import time
from datetime import datetime
import re
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from typing import Union
from starlette.types import Message
from . import config
from .utils import log, profiler, json_utils
from .routers import template, image, shop
import uuid
import gzip

log.setup_log_unicorn()
logger = log.get_logger(__name__)

async def profiler_middleware(request: Request, call_next):
    if request.url.path.startswith('/api'):
        path = re.sub(r'/\d+$', '', request.url.path)
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        profiler.create_profiler_thread(path, request_id)
        try:
            return await call_next(request)
        finally:
            profiler.close_profiler_thread(request_id)
    else:
        return await call_next(request)

async def set_body(request: Request, body: bytes):
    async def receive() -> Message:
        return {'type': 'http.request', 'body': body}
    request._receive = receive

async def read_body(request: Request):
    req_body = b''
    async for chunk in request.stream():
        req_body += chunk
    return req_body

async def read_reponse(response: Response):
    res_body = b''
    async for chunk in response.body_iterator:
        res_body += chunk
    return res_body

def decompress_gzip(data):
    try:
        return str(data, 'utf-8')
    except UnicodeDecodeError:
        decompressed_data = gzip.decompress(data)
        return decompressed_data.decode('utf-8')

async def logging_middleware(request: Request, call_next):
    try:
        start_time = time.time() 
        req_body = await read_body(request)
        await set_body(request, req_body)
        json_req = {}
        if req_body:
            json_req = json_utils.loads(decompress_gzip(req_body))
        response = await call_next(request)

        res_headers = dict(response.headers)
        json_res = {'error': 0}
        if 'content-type' in res_headers and res_headers['content-type'] == 'application/json':
            res_body = await read_reponse(response)
            if res_body:
                try:
                    json_res = json_utils.loads(decompress_gzip(res_body))
                except Exception as e:
                    logger.exception(e)
        data = []
        data.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        data.append(request.method)
        data.append(request.url.path)
        data.append(round((time.time() - start_time) * 1000))
        data.append(dict(**dict(request.query_params), **request.path_params.copy()))
        data.append(json_req)
        print()
        data.append(json_res['error'] if 'error' in json_res else -2)
        data.append(json_res)
        data.append(response.status_code)
        logger.info('\t'.join([str(x) for x in data]))

        if 'content-type' in res_headers and res_headers['content-type'] == 'application/json':
            return Response(content=res_body, status_code=response.status_code, headers=res_headers, media_type=response.media_type)
        else:
            return response
    except Exception as e:
        logger.exception(e)
    
async def athen_middleware(request: Request, call_next):
    if config.ENV != 'local' and not 'static' in request.url.path and '/profilers' != request.url.path:
        api_key = request.headers.get("Authorization")
        # print(api_key)
        if api_key is None or api_key != config.API_KEY:
            return JSONResponse(status_code=403, content='Forbidden')
    return await call_next(request)

async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(f'{process_time:0.4f} sec')
    return response

app = FastAPI()
app.add_middleware(GZipMiddleware)
app.middleware("http")(add_process_time_header)
app.middleware("http")(profiler_middleware)
app.middleware("http")(athen_middleware)
app.middleware("http")(logging_middleware)

app.include_router(template.router)
app.include_router(image.router)
app.include_router(shop.router)
app.mount("/static", StaticFiles(directory="resources/static"), name="static")

# import uvicorn
# if __name__=='__main__':
#     uvicorn.run('main:app', reload=True)