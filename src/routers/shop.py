from fastapi import APIRouter, Request
from .. import api_response, config
from ..utils import log, download
from ..models.view_classifier import ViewClassifier
from threading import Lock

logger = log.get_logger(__name__)
router = APIRouter(prefix="/api/shop")
labels = ['CLOSED', 'BLANK', 'OK']
models = {
    'view': ViewClassifier(config.VIEW_MODEL, labels) if not config.ENV == 'local' else None
}
lock = Lock()

@router.post("/classify")
def detect(request: Request, body: dict):
    try:
        business = body['business']
        image_url = body['image_url']
        if 'view' not in models:
            with lock:
                if 'view' not in models:
                    model = ViewClassifier(config.VIEW_MODEL, labels)
                    models['view'] = model
        if 'view' not in models:
            return api_response.FAIL
        model = models['view']
        img = download.download_image(image_url)
        logits = model.predict(img)
        predicted_label = logits.argmax(-1).item()
        # print(logits[0][predicted_label].numpy())
        conf = logits[0][predicted_label].item()
        data = {'label': labels[predicted_label], 'conf': conf}
        resp = api_response.get_success_resp()
        resp['data'] = data
        return resp
    except Exception as e:
        logger.exception(e)
        resp = api_response.UNKNOW_EXCEPTION
        return resp