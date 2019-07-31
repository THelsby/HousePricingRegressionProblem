import pip
from pip._internal import main as pipmain


def install_whl(path):
    pipmain(['install', path])


install_whl('PricePrediction/xgboost-0.90-cp37-cp37m-win_amd64.whl')
