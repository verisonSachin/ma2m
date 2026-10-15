"""Lineart isolation, screentone suppression, and anime contour preprocessor."""

from typing import Union, Tuple
import numpy as np
import cv2
from PIL import Image


class LineartCleaner:
    """Preprocesses raw manga panels to remove screentones and isolate crisp lineart."""

    def __init__(
        self,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
        xdog_gamma: float = 0.98,
        xdog_phi: float = 200.0,
        xdog_epsilon: float = 0.01,
        xdog_sigma: float = 0.8,
        xdog_k: float = 1.6,
    ):
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.xdog_gamma = xdog_gamma
        self.xdog_phi = xdog_phi
        self.xdog_epsilon = xdog_epsilon
        self.xdog_sigma = xdog_sigma
        self.xdog_k = xdog_k

    def clean(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes manga panel to yield:
        1. Descreentoned tone-smoothed grayscale/RGB base.
        2. Isolated anime lineart (black ink lines on pure white canvas).

        Returns:
            smoothed_base: RGB uint8 array with halftone dots eliminated.
            lineart_anime: RGB uint8 array (3 channels, 0-255) ideal for ControlNet Lineart.
        """
        img_rgb = self._load_rgb(image_input)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Bilateral filter: removes high-frequency halftone dot grids while maintaining step edges
        smoothed_gray = cv2.bilateralFilter(
            gray,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space,
        )

        # Additional median blur pass to clear fine stippling
        smoothed_gray = cv2.medianBlur(smoothed_gray, 3)

        # 2. XDoG (Extended Difference of Gaussians) for crisp manga line extraction
        lineart_gray = self._xdog(
            smoothed_gray,
            sigma=self.xdog_sigma,
            k=self.xdog_k,
            gamma=self.xdog_gamma,
            phi=self.xdog_phi,
            eps=self.xdog_epsilon,
        )

        # Normalize lineart to high-contrast ink
        lineart_gray = cv2.normalize(lineart_gray, None, 0, 255, cv2.NORM_MINMAX)

        # Morphological thinning if lines are excessively thick
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        lineart_gray = cv2.morphologyEx(lineart_gray, cv2.MORPH_CLOSE, kernel)

        smoothed_base = cv2.cvtColor(smoothed_gray, cv2.COLOR_GRAY2RGB)
        lineart_rgb = cv2.cvtColor(lineart_gray, cv2.COLOR_GRAY2RGB)

        return smoothed_base, lineart_rgb

    def _xdog(
        self,
        image: np.ndarray,
        sigma: float,
        k: float,
        gamma: float,
        phi: float,
        eps: float,
    ) -> np.ndarray:
        """
        Extended Difference of Gaussians (XDoG) operator.
        Preserves manga pencil and ink strokes while dropping tone gradients.
        """
        img_float = image.astype(np.float32) / 255.0

        # Two Gaussian blurs with different standard deviations
        g1 = cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
        g2 = cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma * k, sigmaY=sigma * k)

        # Difference with scaling factor gamma
        diff = g1 - gamma * g2

        # Soft thresholding using tanh
        output = np.zeros_like(diff)
        condition = diff >= eps
        output[condition] = 1.0
        output[~condition] = 1.0 + np.tanh(phi * (diff[~condition] - eps))

        output_uint8 = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        return output_uint8

    def _load_rgb(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image_input, str):
            bgr = cv2.imread(image_input)
            if bgr is None:
                raise ValueError(f"Cannot read image: {image_input}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            return image_input
        raise TypeError(f"Invalid input type: {type(image_input)}")
