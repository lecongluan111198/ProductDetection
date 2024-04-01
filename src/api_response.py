from enum import Enum

class ERROR_CODE(Enum):
    SUCCESS = 0
    FAIL = -1
    EXCEPTION = -2
    DENIED = -3
    NOT_FOUND = -4
    INVALID_PARAMS = -4

FAIL = {"error": ERROR_CODE.FAIL.value, "message": "An error occurred"}
INVALID_PARAMS = {"error": ERROR_CODE.INVALID_PARAMS.value, "message": "Invalid params"}
NOT_FOUND = {"error": ERROR_CODE.NOT_FOUND.value, "message": "Not found"}
UNKNOW_EXCEPTION = {"error": ERROR_CODE.EXCEPTION.value, "message": "An error occurred"}

def createResp(error: ERROR_CODE, message):
    return {"error": error.value, "message": message}

def get_success_resp():
    return {"error": ERROR_CODE.SUCCESS.value, "message": "Success"}