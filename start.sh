#!/bin/sh
# conda activate kdfdetect
ENV=local uvicorn src.main:app --host 0.0.0.0 --port 9033 --workers 1 --reload
