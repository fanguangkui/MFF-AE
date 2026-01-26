# from DetectOutlier.utlis.myutils import Utils
# import numpy as np

#add the baselines from the pyod package
from .baseline.iforest import IForest
from .baseline.ocsvm import OCSVM
from .baseline.abod import ABOD
from .baseline.copod import COPOD
from .baseline.ecod import ECOD
from .baseline.hbos import HBOS
from .baseline.knn import KNN
from .baseline.lmdd import LMDD
from .baseline.loda import LODA
from .baseline.lof import LOF
from .baseline.mad import MAD
from .baseline.mcd import MCD
from .baseline.pca import PCA
from .baseline.rod import ROD
from .baseline.sod import SOD
from .baseline.dif import DIF
from .baseline.gmm import GMM


def ModelFactory(model_name, model_para):
    '''
    :param seed: seed for reproducible results
    :param model_name: model name
    :param tune: if necessary, tune the hyper-parameter based on the validation set constructed by the labeled anomalies
    '''

    model_dict = {
        'IsolationForest':IForest,
        'IForest':IForest,
        'OCSVM':OCSVM,
        'ABOD':ABOD,
        'COPOD':COPOD,
        'ECOD':ECOD,
        'HBOS':HBOS,
        'KNN':KNN,
        'LMDD':LMDD,
        'LODA':LODA,
        'LOF':LOF,
        'MAD':MAD,
        'MCD':MCD,
        'PCA':PCA,
        'ROD':ROD,
        'SOD':SOD,
        'DIF': DIF,
        'GMM': GMM,
    }
    if model_name in model_dict:
        return model_dict[model_name](**model_para)
    else:
        raise ValueError("no model in model.PyOD file")
