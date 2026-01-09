# from DetectOutlier.utlis.myutils import Utils
# import numpy as np

#add the baselines from the pyod package
from .baseline.alad import ALAD
from .baseline.iforest import IForest
from .baseline.ocsvm import OCSVM
from .baseline.abod import ABOD
from .baseline.cblof import CBLOF
from .baseline.cof import COF
from .baseline.combination import aom
from .baseline.copod import COPOD
from .baseline.ecod import ECOD
from .baseline.feature_bagging import FeatureBagging
from .baseline.hbos import HBOS
from .baseline.knn import KNN
from .baseline.lmdd import LMDD
from .baseline.loda import LODA
from .baseline.lof import LOF
from .baseline.loci import LOCI
from .baseline.lscp import LSCP
from .baseline.mad import MAD
from .baseline.mcd import MCD
from .baseline.pca import PCA
from .baseline.rod import ROD
from .baseline.sod import SOD
from .baseline.sos import SOS
from .baseline.vae import VAE
from .baseline.vae_torch import VAE_Pytorch
from .baseline.mlm_ve_torch import MLM_AE
from .baseline.Local_vae_torch import LocalVAE
from .baseline.auto_encoder_torch import AutoEncoder
from .baseline.adaptive_auto_encoder_torch import AdaptiveAE
from .baseline.ae1svm import AE1SVM
from .baseline.dif import DIF
from .baseline.gmm import GMM
from .baseline.anogan import AnoGAN
from .baseline.adaptive_residual_auto_encoder_torch import AdaptiveResidualAE


# from .baseline.MaskCTR.MaskCTRAE import MaskCTRAE
# from .baseline.MaskCTR.MaskCTRAE2 import MaskCTRAE2
# from .baseline.MaskCTR.Mask2CTRAE import Mask2CTRAE
# from .baseline.MaskCTR.MaskCTR_AEScore import MaskCTR_AEScore
# from .baseline.MaskCTR.MaskCTRMemoryCE import MaskCTRMemoryCE
# from .baseline.MaskCTR.MaskCTRMemoryCEScoreDist import MaskCTRMemoryCEScoreDist
# from .baseline.MaskCTR.MaskCTRMemoryCountCE import MaskCTRMemoryCountCE
# from .baseline.MaskCTR.MUSEATTENCTRMemoryCE import MUSEAttenCTRMemoryCE

from .baseline.so_gaal import SO_GAAL
from .baseline.so_gaal_pytorch import SOGAAL_Pytorch
from .baseline.mo_gaal import MO_GAAL
from .baseline.xgbod import XGBOD
from .baseline.deep_svdd import DeepSVDD


def ModelFactory(model_name, model_para):
    '''
    :param seed: seed for reproducible results
    :param model_name: model name
    :param tune: if necessary, tune the hyper-parameter based on the validation set constructed by the labeled anomalies
    '''

    model_dict = {
        'IForest':IForest,
        'ALAD':ALAD,
        'OCSVM':OCSVM,
        'ABOD':ABOD,
        'CBLOF':CBLOF,
        'COF':COF,
        'AOM':aom,
        'COPOD':COPOD,
        'ECOD':ECOD,
        'FeatureBagging':FeatureBagging,
        'HBOS':HBOS,
        'KNN':KNN,
        'LMDD':LMDD,
        'LODA':LODA,
        'LOF':LOF,
        'LOCI':LOCI,
        'LSCP':LSCP,
        'MAD':MAD,
        'MCD':MCD,
        'PCA':PCA,
        'ROD':ROD,
        'SOD':SOD,
        'SOS':SOS,
        # 'CNNAE':CNNAE,
        'VAE':VAE,
        'MLM_AE':MLM_AE,
        'VAE_Pytorch':VAE_Pytorch,
        'LocalVAE':LocalVAE,
        'DeepSVDD': DeepSVDD,
        'AutoEncoder': AutoEncoder,
        'AdaptiveAE': AdaptiveAE,
        'DIF': DIF,
        'GMM': GMM,
        'AdaptiveResidualAE': AdaptiveResidualAE,
        # 'AdaptiveAEScore': AdaptiveAEScore,
        # 'AdaptiveMaskAEScore': AdaptiveMaskAEScore,
        # 'MaskCTRMemoryCE': MaskCTRMemoryCE,
        # 'MaskCTRMemoryCEScoreDist': MaskCTRMemoryCEScoreDist,
        # 'MaskCTRMemoryCountCE': MaskCTRMemoryCountCE,
        # 'MUSEAttenCTRMemoryCE': MUSEAttenCTRMemoryCE,
        # 'MaskCTRAE': MaskCTRAE,
        # 'MaskCTRAE2': MaskCTRAE2,
        # 'Mask2CTRAE': Mask2CTRAE,
        # 'MaskCTR_AEScore': MaskCTR_AEScore,
        'AE1SVM': AE1SVM,
        'SOGAAL': SO_GAAL,
        'MOGAAL': MO_GAAL,
        'XGBOD': XGBOD,
        "SOGAAL_Pytorch": SOGAAL_Pytorch,
    }
    if model_name in model_dict:
        return model_dict[model_name](**model_para)
    else:
        raise "no model in model.PyOD file"
