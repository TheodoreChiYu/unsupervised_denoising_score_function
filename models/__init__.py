from typing import Optional

from tools.model import BaseModel
from tools.config import Config
from tools.logger import Logger
from tools.optimizer import Scaler

from models.gaussian_score_func import GaussianScoreFuncModel
from models.gaussian_map_score_func import GaussianMapScoreFuncModel
from models.gaussian_conv_score_func import GaussianConvScoreFuncModel
from models.gaussian_map_conv_score_func import GaussianMapConvScoreFuncModel
from models.poisson_score_func import PoissonScoreFuncModel
from models.poisson_conv_score_func import PoissonConvScoreFuncModel
from models.poisson_gaussian_score_func import PoissonGaussianScoreFuncModel
from models.poisson_conv_gaussian_score_func import \
    PoissonConvGaussianScoreFuncModel
from models.gamma_score_func import GammaScoreFuncModel
from models.gamma_conv_score_func import GammaConvScoreFuncModel
from models.gamma_gaussian_score_func import GammaGaussianScoreFuncModel
from models.gamma_conv_gaussian_score_func import \
    GammaConvGaussianScoreFuncModel
from models.rayleigh_score_func import RayleighScoreFuncModel
from models.rayleigh_conv_score_func import RayleighConvScoreFuncModel
from models.rayleigh_gaussian_score_func import RayleighGaussianScoreFuncModel
from models.rayleigh_conv_gaussian_score_func import \
    RayleighConvGaussianScoreFuncModel


__ALL_MODELS__ = {
    "Gaussian": GaussianScoreFuncModel,
    "GaussianMap": GaussianMapScoreFuncModel,
    "GaussianConv": GaussianConvScoreFuncModel,
    "GaussianMapConv": GaussianMapConvScoreFuncModel,
    "Poisson": PoissonScoreFuncModel,
    "PoissonConv": PoissonConvScoreFuncModel,
    "PoissonGaussian": PoissonGaussianScoreFuncModel,
    "PoissonConvGaussian": PoissonConvGaussianScoreFuncModel,
    "Gamma": GammaScoreFuncModel,
    "GammaConv": GammaConvScoreFuncModel,
    "GammaGaussian": GammaGaussianScoreFuncModel,
    "GammaConvGaussian": GammaConvGaussianScoreFuncModel,
    "Rayleigh": RayleighScoreFuncModel,
    "RayleighConv": RayleighConvScoreFuncModel,
    "RayleighGaussian": RayleighGaussianScoreFuncModel,
    "RayleighConvGaussian": RayleighConvGaussianScoreFuncModel,

}


def build_model(model_type: str, model_config: Config,
                scaler: Optional[Scaler], logger: Logger) -> BaseModel:
    return __ALL_MODELS__[model_type](model_config, scaler, logger)
