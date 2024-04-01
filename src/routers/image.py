import torch
from typing import List
from fastapi import APIRouter, Request
from .. import api_response, config
from ..utils import log, pytorch_loader, download, box_utils, profiler
from ..models.bottle_detector import BottleDetector
import re
from threading import Lock

logger = log.get_logger(__name__)
router = APIRouter(prefix="/api/image")

version_tac_bottle = re.search(r'([^/]+)\.pt$', config.TAC_BOTTLE_MODEL).group(1)
version_tac_window = re.search(r'([^/]+)\.pt$', config.TAC_WINDOW_MODEL).group(1)
version_banh = re.search(r'([^/]+)\.pt$', config.BANH_MODEL).group(1)
models = {}
lock = {
    'tac': Lock(),
    'banh': Lock()
}

@router.post("/detect")
def detect(request: Request, body: dict):
    try:
        business = body['business']
        image_url = body['image_url']
        if business == 2:
            return process_tac(business, image_url, request.state.request_id)
        elif business == 4:
            return process_banh(business, image_url)
        else:
            return api_response.DENIED
    except Exception as e:
        logger.exception(e)
        resp = api_response.UNKNOW_EXCEPTION
        return resp

def process_tac(business, image_url, request_id = None):
    if 'tac' not in models:
        with lock['tac']:
            if 'tac' not in models:
                model_tac = BottleDetector(config.TAC_BOTTLE_MODEL)
                models['tac'] = model_tac
            if 'window' not in models:
                model_window = BottleDetector(config.TAC_WINDOW_MODEL)
                models['window'] = model_window
    if 'tac' not in models or 'window' not in models:
        return api_response.FAIL
    model_tac = models['tac']
    model_window = models['window']
    version = [f'dau_{version_tac_bottle}', f'window_{version_tac_window}']
    retry = 3
    while retry > 0:
        retry -= 1
        try:
            img = download.download_image(image_url)
            data = {}
            try:
                profiler.push('predict_bottle', request_id)
                group_cnt, labels = model_tac.predict(img)
            finally:
                profiler.pop('predict_bottle', request_id)
            data['product'] = group_cnt
            data['layout'] = box_utils.build_layout(labels, request_id = request_id)
            try:
                profiler.push('predict_window', request_id)
                data['window_frame'], _ = model_window.predict(img)
            finally:
                profiler.pop('predict_window', request_id)
            resp = api_response.get_success_resp()
            resp['data'] = data
            resp['version'] = version
            return resp
        except Exception as e:
            if retry <= 0:
                raise e
    return api_response.FAIL

def process_banh(business, image_url):
    if 'banh' not in models:
        with lock['banh']:
            if 'banh' not in models:
                model_banh = BottleDetector(config.BANH_MODEL)
                models['banh'] = model_banh
    if 'banh' not in models:
        return api_response.FAIL
    
    model_banh = models['banh']
    version = [f'banh_{version_banh}']
    img = download.download_image(image_url)
    retry = 3
    while retry > 0:
        retry -= 1
        try:
            data = {}
            group_cnt, _ = model_banh.predict(img)
            data['product'] = group_cnt
            resp = api_response.get_success_resp()
            resp['data'] = data
            resp['version'] = version
            return resp
        except Exception as e:
            if retry <= 0:
                raise e
    return api_response.FAIL
