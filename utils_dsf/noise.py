import numpy as np

from tools.config import Config


NOISE_MODELS = [
    "gaussian", "gaussian_map", "gaussian_conv", "gaussian_map_conv",
    "poisson", "gamma", "rayleigh",
    "poisson-gaussian", "gamma-gaussian", "rayleigh-gaussian",
    "poisson-conv", "gamma-conv", "rayleigh-conv",
    "poisson-conv-gaussian", "gamma-conv-gaussian", "rayleigh-conv-gaussian",
]


def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.randn(*image.shape) * sigma
    noisy_image = image + noise
    return noisy_image


def add_gaussian_map_noise(image: np.ndarray, a: float, b: float) -> np.ndarray:
    noise = np.random.randn(*image.shape)
    noisy_image = image + (a + b * image) * noise
    return noisy_image


def add_poisson_noise(image: np.ndarray, lambda_: float) -> np.ndarray:
    noisy_image = np.random.poisson(lambda_ * image) / lambda_
    return noisy_image


def add_gamma_noise(image: np.ndarray, alpha: float) -> np.ndarray:
    eta = np.random.gamma(alpha, 1, size=image.shape) / alpha
    noisy_image = eta * image
    return noisy_image


def add_rayleigh_noise(image: np.ndarray, delta: float) -> np.ndarray:
    eta = np.random.rayleigh(delta, size=image.shape)
    eta = eta + 1.
    noisy_image = eta * image
    return noisy_image


def add_conv(image: np.ndarray, conv_3x3_filter: np.ndarray) -> np.ndarray:
    space_kernel = np.zeros(image.shape[:2])
    space_kernel[:2, :2] = conv_3x3_filter[-2:, -2:]
    space_kernel[-1:, -1:] = conv_3x3_filter[:1, :1]
    space_kernel[:2, -1:] = conv_3x3_filter[-2:, :1]
    space_kernel[-1:, :2] = conv_3x3_filter[:1, -2:]
    freq_kernel = np.fft.fft2(space_kernel)

    if len(image.shape) == 3:
        freq_kernel = freq_kernel[..., None]

    freq = np.fft.fft2(image, axes=[0, 1])
    conv_freq = freq_kernel * freq
    conv_image = np.fft.ifft2(conv_freq, axes=[0, 1])
    conv_image = conv_image.real

    return conv_image


def add_gaussian_conv_noise(image: np.ndarray, sigma: float,
                            conv_3x3_filter: np.ndarray) -> np.ndarray:
    noise = np.random.randn(*image.shape) * sigma
    conv_noise = add_conv(noise, conv_3x3_filter)
    noisy_image = image + conv_noise
    return noisy_image


def add_gaussian_map_conv_noise(image: np.ndarray, a: float, b: float,
                                conv_3x3_filter: np.ndarray) -> np.ndarray:
    noise = np.random.randn(*image.shape)
    noise = (a + b * image) * noise
    conv_noise = add_conv(noise, conv_3x3_filter)
    noisy_image = image + conv_noise
    return noisy_image


def add_noise(image: np.ndarray, noise_type: str,
              noise_config: Config) -> np.ndarray:
    if noise_type == "gaussian":
        noisy_image = add_gaussian_noise(image, noise_config.sigma)
    elif noise_type == "gaussian_map":
        noisy_image = add_gaussian_map_noise(image, noise_config.a,
                                             noise_config.b)
    elif noise_type == "gaussian_conv":
        noisy_image = add_gaussian_conv_noise(image, noise_config.sigma,
                                              noise_config.conv_3x3_filter)
    elif noise_type == "gaussian_map_conv":
        noisy_image = add_gaussian_map_conv_noise(
            image, noise_config.a, noise_config.b, noise_config.conv_3x3_filter
        )
    elif noise_type == "poisson":
        noisy_image = add_poisson_noise(image, noise_config.lambda_)
    elif noise_type == "gamma":
        noisy_image = add_gamma_noise(image, noise_config.alpha)
    elif noise_type == "rayleigh":
        noisy_image = add_rayleigh_noise(image, noise_config.delta)
    elif noise_type == "conv":
        noisy_image = add_conv(image, noise_config.conv_3x3_filter)
    else:
        raise ValueError("noise_type is wrong.")

    return noisy_image


def get_noisy_image(image: np.ndarray, noise_config: Config):
    """
    Crop the center part of image such that height and width are 32x.

    :param image: np.ndarray with shape of (h, w, c).
    :param noise_config: Config, contains noise model and its params.
    :return: np.ndarray with shape of (h, w, c).
    """
    assert noise_config.noise_model in NOISE_MODELS

    noisy_image = image
    noise_types = noise_config.noise_model.split("-")
    for noise_type in noise_types:
        noisy_image = add_noise(noisy_image, noise_type, noise_config)

    return noisy_image
