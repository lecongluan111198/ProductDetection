import os
from dotenv import load_dotenv

env = os.environ.get('ENV')
if env is None:
    env = 'production'
print("Load config from " + env + ".env")
load_dotenv("conf/" + env + ".env")

ENV = env
PROJECT_NAME = os.getenv('PROJECT_NAME')
THRESH_HOLD = float(os.getenv('THRESH_HOLD'))
TAC_BOTTLE_MODEL = os.getenv('TAC_BOTTLE_MODEL')
TAC_WINDOW_MODEL = os.getenv('TAC_WINDOW_MODEL')
BANH_MODEL = os.getenv('BANH_MODEL')
VIEW_MODEL = os.getenv('VIEW_MODEL')
API_KEY = 'Bearer o5KcRbLoA8gZOD50UpzAJoVRnYjxyg'

CLS_2_GROUP = {
    'Cook0.4L': 'COOK_0.4L',
    'Cook1L': 'COOK_1L',
    'Cook2L': 'COOK_2L',
    'Cook5L': 'COOK_5L',
    'CookNutri1L': 'COOKNT_1L',
    'CookVitamin1L': 'COOKNT_1L',
    'CookVitamin2L': 'COOKNT_2L',
    'CookVitamin5L': 'COOKNT_5L',
    'Gold1L': 'TAG_1L',
    'Gold2L': 'TAG_2L',
    'Gold5L': 'TAG_5L',
    'HatCai1L': 'CAI_25KG',
    'HuongDuong1L': 'HD_1L',
    'KeGold': 'KeGold',
    'KeMarvela': 'KeMarvela',
    'KeNanh': 'KeNanh',
    'KeNanh2Tang': 'KeNanh2Tang',
    'KeVio': 'KeVio',
    'Lut1L': 'GAO_LUT_1L',
    'MarvelaNanh1L': 'MAR_NA_1L',
    'MarvelaNanh2L': 'MAR_NA_2L',
    'MarvelaNanh5L': 'MAR_NA_5L',
    'MarvelaNutri1L': 'MAR_DD_1L',
    'MarvelaNutri2L': 'MAR_DD_2L',
    'Me0.4L': 'ME_0.4L',
    'Me1L': 'ME_1L',
    'Nanh1L': 'DAU_NA_1L',
    'Nanh2L': 'DAU_NA_2L',
    'Nanh5L': 'DAU_NA_5L',
    'Ngon0.4L': 'NGON_0.37L',
    'Ngon1.8L': 'NGON_1.8L',
    'Ngon1L': 'NGON_0.88L',
    'Ngon5L': 'NGON_4.7L',
    'Olita1L': 'OLITA_1L',
    'Olita2L': 'OLITA_2L',
    'Olita5L': 'OLITA_5L',
    'OlitaDo1L': 'OLITA_1L',
    'OlitaDo2L': 'OLITA_2L',
    'OlitaDo5L': 'OLITA_5L',
    'Phong1L': 'PHONG_1L',
    'SEASON1L': 'SEASON_1L',
    'VanTho0.4L': 'VTHO_0.4L',
    'VanTho1L': 'VTHO_1L',
    'VanTho2L': 'VTHO_2L',
    'VanTho5L': 'VTHO_5L',
    'VioDo': 'VIOG_0.25L',
    'VioTrang': 'VIOM_0.25L',
    'VioXanh': 'VIOO_0.25L',
    'VanThoCC1L': 'VTHOCC_1L',
    'VanThoCC2L': 'VTHOCC_2L',
    'VanThoCC5L': 'VTHOCC_5L',
    'VanPhuc1L': 'VANPHUC_0.75L',
    'MeThom0.1L': 'METHO_0.1L',
    'MeThom0.25L': 'METHO_0.25L',
    'Olive0.25L': 'OLIVE_0.25L'
}