"""Open-source & free image generation API clients (Pollinations.ai, HuggingFace, Local Docker SD-WebUI/ComfyUI)."""

import os
import io
import base64
import urllib.parse
from typing import Optional, Tuple, Dict, Any
import requests
import numpy as np
from PIL import Image
import cv2


class FreeImageAPIClient:
    """Client for generating or colorizing anime frames using free/open-source external or containerized APIs."""

    def __init__(
        self,
        provider: str = os.getenv("IMAGE_API_PROVIDER", "pollinations"),  # "pollinations", "huggingface", "local_sd_api"
        local_api_url: str = os.getenv("LOCAL_SD_API_URL", "http://sd-api:7860"),
        hf_api_token: Optional[str] = os.getenv("HF_TOKEN", None),
        hf_model: str = os.getenv("HF_MODEL", "cagliostrolab/animagine-xl-3.1"),
        timeout: int = 60,
    ):
        self.provider = provider.lower()
        self.local_api_url = local_api_url.rstrip("/")
        self.hf_api_token = hf_api_token
        self.hf_model = hf_model
        self.timeout = timeout

    def generate_or_colorize(
        self,
        prompt: str,
        lineart_rgb: Optional[np.ndarray] = None,
        negative_prompt: str = "lowres, text, watermark, bad quality, blurry",
        width: int = 512,
        height: int = 768,
        seed: int = 42,
    ) -> Optional[np.ndarray]:
        """
        Send image generation request to selected open API provider.

        Returns:
            RGB numpy array or None on failure.
        """
        if self.provider == "pollinations":
            return self._call_pollinations(prompt, negative_prompt, width, height, seed)
        elif self.provider == "huggingface":
            return self._call_huggingface(prompt, negative_prompt, width, height, seed)
        elif self.provider in ("local_sd_api", "docker_api", "a1111"):
            return self._call_local_sd_api(prompt, lineart_rgb, negative_prompt, width, height, seed)
        else:
            print(f"[FreeImageAPIClient] Unknown provider '{self.provider}'. Attempting pollinations fallback.")
            return self._call_pollinations(prompt, negative_prompt, width, height, seed)

    def _call_pollinations(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
    ) -> Optional[np.ndarray]:
        """
        Pollinations.ai is an open-source, free, public generative AI endpoint requiring no API key.
        """
        try:
            # Enhance prompt with anime styling tags
            full_prompt = f"masterpiece, best quality anime illustration, {prompt}"
            encoded_prompt = urllib.parse.quote(full_prompt)

            # Pollinations free API URL format
            url = (
                f"https://image.pollinations.ai/prompt/{encoded_prompt}"
                f"?width={width}&height={height}&seed={seed}&nologo=true&model=flux"
            )

            print(f"[FreeImageAPIClient] Querying free Pollinations.ai API for anime generation...")
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                img_pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
                img_np = np.array(img_pil)
                if (img_np.shape[0], img_np.shape[1]) != (height, width):
                    img_np = cv2.resize(img_np, (width, height), interpolation=cv2.INTER_LANCZOS4)
                return img_np
            else:
                print(f"[FreeImageAPIClient] Pollinations API returned status code {resp.status_code}")
                return None
        except Exception as e:
            print(f"[FreeImageAPIClient] Pollinations API request failed: {e}")
            return None

    def _call_huggingface(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
    ) -> Optional[np.ndarray]:
        """Free Hugging Face Serverless Inference API."""
        try:
            api_url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
            headers = {}
            if self.hf_api_token:
                headers["Authorization"] = f"Bearer {self.hf_api_token}"

            payload = {
                "inputs": f"anime art, {prompt}",
                "parameters": {
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "guidance_scale": 7.0,
                },
            }

            print(f"[FreeImageAPIClient] Querying Hugging Face Inference API ({self.hf_model})...")
            resp = requests.post(api_url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                img_pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
                return np.array(img_pil)
            else:
                print(f"[FreeImageAPIClient] Hugging Face API returned status {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            print(f"[FreeImageAPIClient] Hugging Face API error: {e}")
            return None

    def _call_local_sd_api(
        self,
        prompt: str,
        lineart_rgb: Optional[np.ndarray],
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
    ) -> Optional[np.ndarray]:
        """
        Call a local container running Stable Diffusion WebUI or ComfyUI with standard REST API.
        Endpoint: /sdapi/v1/img2img (with ControlNet) or /sdapi/v1/txt2img.
        """
        try:
            # Prepare lineart as base64 if available
            init_images = []
            controlnet_args = []

            if lineart_rgb is not None:
                _, buffer = cv2.imencode(".png", cv2.cvtColor(lineart_rgb, cv2.COLOR_RGB2BGR))
                b64_lineart = base64.b64encode(buffer).decode("utf-8")
                init_images.append(b64_lineart)
                controlnet_args = [
                    {
                        "input_image": b64_lineart,
                        "module": "lineart_anime",
                        "model": "control_v11p_sd15_lineart_anime",
                        "weight": 1.0,
                    }
                ]

            endpoint = f"{self.local_api_url}/sdapi/v1/img2img" if init_images else f"{self.local_api_url}/sdapi/v1/txt2img"

            payload: Dict[str, Any] = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": 25,
                "cfg_scale": 7.5,
                "width": width,
                "height": height,
                "seed": seed,
            }
            if init_images:
                payload["init_images"] = init_images
                payload["denoising_strength"] = 0.75
            if controlnet_args:
                payload["alwayson_scripts"] = {
                    "controlnet": {
                        "args": controlnet_args
                    }
                }

            print(f"[FreeImageAPIClient] Querying local SD API container at {endpoint}...")
            resp = requests.post(endpoint, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if "images" in data and len(data["images"]) > 0:
                    raw_b64 = data["images"][0]
                    img_bytes = base64.b64decode(raw_b64.split(",", 1)[-1])
                    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    return np.array(img_pil)

            print(f"[FreeImageAPIClient] Local SD API returned: {resp.status_code}")
            return None
        except Exception as e:
            print(f"[FreeImageAPIClient] Local SD API connection failed: {e}")
            return None
