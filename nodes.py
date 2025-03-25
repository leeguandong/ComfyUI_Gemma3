import os
import torch
import numpy as np
import comfy.model_management as mm
from PIL import Image
import folder_paths
from pathlib import Path
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


class Gemma3ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_id": (
                ["google/gemma-3-27b-it", "google/gemma-3-12b-it", "google/gemma-3-1b-it", "google/gemma-3-4b-it"],
                {"default": "google/gemma-3-27b-it"}),
                "load_local_model": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "local_gemma3_model_path": ("STRING", {"default": "google/gemma-3-27b-it"}),
            }
        }

    RETURN_TYPES = ("MODEL", "PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "gemma3"

    def load_model(self, model_id, load_local_model, *args, **kwargs):
        device = mm.get_torch_device()
        if load_local_model:
            # 如果加载本地模型，直接使用用户提供的路径
            model_id = kwargs.get("local_gemma3_model_path", model_id)
        else:
            # 如果加载 Hugging Face 模型，下载到 ComfyUI 的模型目录
            gemma_dir = os.path.join(folder_paths.models_dir, "Gemma3")
            os.makedirs(gemma_dir, exist_ok=True)

            # 下载模型到指定目录
            model_id = "google/gemma-3-27b-it"
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id, cache_dir=gemma_dir, device_map="auto"
            ).eval().to(device)
            processor = AutoProcessor.from_pretrained(
                model_id, cache_dir=gemma_dir
            )
            return (model, processor)

        # 加载本地模型
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        ).eval().to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        return (model, processor)


class ApplyGemma3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "processor": ("PROCESSOR",),
                "prompt": ("STRING", {"default": "Describe this image in detail."}),
                "max_new_tokens": ("INT", {"default": 100, "min": 1, "max": 1000}),
            },
            "optional": {
                "image": ("IMAGE",),  # 图像输入作为可选项
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "apply_gemma3"
    CATEGORY = "gemma3"

    def apply_gemma3(self, model, processor, prompt, max_new_tokens, image=None):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            }
        ]

        # 如果有图像输入，将图像和提示添加到消息中
        if image is not None:
            image_pil = tensor2pil(image)  # 将 ComfyUI 的 IMAGE 格式转换为 PIL.Image
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": prompt}
                ]
            })
        else:
            # 如果没有图像输入，仅使用文本提示
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            })

        # 处理输入
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        # 生成文本
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)
        return (decoded,)


NODE_CLASS_MAPPINGS = {
    "Gemma3ModelLoader": Gemma3ModelLoader,
    "ApplyGemma3": ApplyGemma3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemma3ModelLoader": "Gemma3 Model Loader",
    "ApplyGemma3": "Apply Gemma3",
}
