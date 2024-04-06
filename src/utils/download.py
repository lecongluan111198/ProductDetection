import cv2
import requests
import numpy as np
import os

# proxies = {
#   'http': os.getenv('PROXY_HTTP'),
#   'https': os.getenv('PROXY_HTTPS')
# }

def download_image(url):
  local = not (url.startswith("http://") or url.startswith("https://"))
  url = url.replace('\n', '').replace(' ', '')
  if local:
    url = url
    img = cv2.imread(url, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
  else:
    # resp = requests.get(url, proxies=proxies, timeout=10)
    resp = requests.get(url, timeout=10)
    img = np.asarray(bytearray(resp.content), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
