import importlib
import logging
from enum import Enum

import torch

from PIL import Image, ImageFilter
import cv2
import os
import copy
import numpy as np
from typing import Optional, Type
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image, AutoPipelineForInpainting
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline
from diffusers import StableDiffusionControlNetInpaintPipeline, StableDiffusionXLControlNetInpaintPipeline, DDIMScheduler
from diffusers.schedulers import KarrasDiffusionSchedulers

from .pipelines.masked_stable_diffusion_img2img import MaskedStableDiffusionImg2ImgPipeline
from .pipelines.masked_stable_diffusion_xl_img2img import MaskedStableDiffusionXLImg2ImgPipeline
from transformers import CLIPProcessor, CLIPTextModel
#from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
# from diffusers import StableDiffusionKDiffusionPipeline

class ModelType(Enum):
    SD = 1
    SDXL = 2


def get_diffusion_scheduler_names():
    """
    return list of schedulers that can be use in our pipelines
    """
    return [scheduler.name for scheduler in KarrasDiffusionSchedulers]


def add_scheduler(pipe, scheduler):
    # setup scheduler
    if scheduler is None:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        return 'DPMSolverMultistepScheduler'
    else:
        # Import the scheduler class dynamically based on the provided name
        try:
            module_name = "diffusers.schedulers"
            class_name = scheduler
            module = importlib.import_module(module_name)
            scheduler_class = getattr(module, class_name)
            pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
            return class_name
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"Invalid scheduler specified {scheduler}") from exc
    return None


class BasePipe:
    """
    Base class for all pipelines.

    Provides some basic functionality to load and save models,
    as well as pipeline configuration
    """
    _class = None
    _classxl = None

    def __init__(self, model_id: str,
                 sd_pipe_class: Optional[Type[DiffusionPipeline]] = None,
                 pipe: Optional[DiffusionPipeline] = None,
                 model_type: Optional[ModelType] = None, device=None, **args):
        """
        Constructor

        Args:
            model_id (str):
                path or id of the model to load
            sd_pipe_class (Type, *optional*):
              a subclass of DiffusionPipeline to load model to.
            pipe (DiffusionPipeline, *optional*):
                an instance of the pipeline to use,
                if provided the model_id won't be used for loading.
            model_type (ModelType, *optional*):
                A flag to selected between SD or SDXL if neither sd_pipe_class nor pipe is given
            **args:
                additional arguments passed to sd_pipe_class constructor
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipe = pipe
        self._scheduler = None
        self._hypernets = []
        self._model_id = os.path.expanduser(model_id)
        self.pipe_params = dict()
        # Creating a stable diffusion pipeine
        args = {**args}
        if 'torch_dtype' not in args:
            if device == torch.device('cpu'):
                args['torch_dtype'] = torch.float32
            else:
                args['torch_dtype'] = torch.float16
        if self.pipe is None:
            self.pipe = self._load_pipeline(sd_pipe_class, model_type, args)
        self._initialize_pipe(device)
            
    def _initialize_pipe(self, device):
        # sometimes text encoder is on a different device
        # if self.pipe.device != device:
        self.pipe.to(device)
        # self.pipe.enable_attention_slicing()
        # self.pipe.enable_vae_slicing()
        self.pipe.vae.enable_tiling()
        # --- the best one and seems to be enough ---
        # self.pipe.enable_sequential_cpu_offload()
        try:
            import xformers
            self.pipe.enable_xformers_memory_efficient_attention() # attention_op=MemoryEfficientAttentionFlashAttentionOp)
        except ImportError as e:
            logging.warning("xformers not found, can't use efficient attention")

    def _load_pipeline(self, sd_pipe_class, model_type, args):
        if sd_pipe_class is None:
            if self._model_id.endswith('.safetensors'):
                if model_type is None:
                    raise RuntimeError(f"model_type is not specified for safetensors file {self._model_id}")
                pipe_class = self._class if model_type == ModelType.SD else self._classxl
                return pipe_class.from_single_file(self._model_id, **args)
            else:
                return self._autopipeline.from_pretrained(self._model_id, **args)
        else:
            if self._model_id.endswith('.safetensors'):
                return sd_pipe_class.from_single_file(self._model_id, **args)
            else:
                return sd_pipe_class.from_pretrained(self._model_id, **args)

    @property
    def scheduler(self):
        """The scheduler used by the pipeline"""
        return self._scheduler

    @property
    def hypernets(self):
        return self._hypernets

    @property
    def model_id(self):
        """The model id used by the pipeline"""
        return self._model_id

    def try_set_scheduler(self, inputs):
        # allow for scheduler overwrite
        scheduler = inputs.get('scheduler', None)
        if scheduler is not None and self.pipe is not None:
            sch_set = add_scheduler(self.pipe, scheduler=scheduler)
            if sch_set:
                self._scheduler = sch_set
            inputs.pop('scheduler')

    def load_lora(self, path, multiplier=1.0):
        self.pipe.load_lora_weights(path)
        if 'cross_attention_kwargs' not in self.pipe_params:
            self.pipe_params['cross_attention_kwargs'] = {}
        self.pipe_params['cross_attention_kwargs']["scale"] = multiplier

    def add_hypernet(self, path, multiplier=None):
        from . hypernet import add_hypernet, Hypernetwork
        hypernetwork = Hypernetwork()
        hypernetwork.load(path)
        self._hypernets.append(path)
        hypernetwork.set_multiplier(multiplier if multiplier else 1.0)
        hypernetwork.to(self.pipe.unet.device)
        hypernetwork.to(self.pipe.unet.dtype)
        add_hypernet(self.pipe.unet, hypernetwork)

    def clear_hypernets(self):
        from . hypernet import clear_hypernets
        clear_hypernets(self.pipe.unet)
        self._hypernets = []

    def get_config(self):
        """
        Return parameters for this model.

        :return: dict
        """
        cfg = {"hypernetworks": self.hypernets }
        cfg.update({"model_id": self.model_id })
        cfg['scheduler'] = dict(self.pipe.scheduler.config)
        cfg['scheduler']['class_name'] = self.pipe.scheduler.__class__.__name__
        cfg.update(self.pipe_params)
        return cfg

    def setup(self, steps=50, clip_skip=0, loras=[], **args):
        """
        Setup pipeline for generation.

        Args:
            steps (int, *optional*):
                number of inference steps
            clip_skip (int, *optional*):
                number of top layers to skip in clip model
            **args (dict, *optional*):
                dict with additional parameters such as scheduler and timestep_spacing,
                other parameters will be ignored.
                scheduler is a scheduler class names from diffusers.schedulers module
                timestep_spacing (`str`, defaults to `"leading"`):
                The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
                Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
            :return: None
        """
        self.pipe_params = { 'num_inference_steps': steps }
        assert clip_skip >= 0
        assert clip_skip <= 10
        self.pipe_params['clip_skip'] = clip_skip
        if 'scheduler' in args:
            self.try_set_scheduler(args)
        if 'timestep_spacing' in args:
            self.pipe.scheduler = self.pipe.scheduler.from_config(self.pipe.scheduler.config, timestep_spacing = args['timestep_spacing'])
            args.pop('timestep_spacing')
        for lora in loras:
            self.load_lora(lora)


class Prompt2ImPipe(BasePipe):
    """
    Base class for all pipelines that take a prompt and return an image.
    """
    _class = StableDiffusionPipeline
    _classxl = StableDiffusionXLPipeline
    _autopipeline = AutoPipelineForText2Image

    def __init__(self, model_id: str,
                 pipe: Optional[StableDiffusionPipeline] = None,
                 lpw=False, **args):
        super().__init__(model_id=model_id, pipe=pipe, **args)
        if lpw:
            # convert to lpw
            if isinstance(self.pipe, self._classxl):
                self.pipe = DiffusionPipeline.from_pipe(self.pipe, custom_pipeline="lpw_stable_diffusion_xl")
            elif isinstance(self.pipe, self._class):
                self.pipe = DiffusionPipeline.from_pipe(self.pipe, custom_pipeline="lpw_stable_diffusion")
            else:
                raise RuntimeError(f"Unexpected model type {type(self.pipe)}")


    def setup(self, width=768, height=768, guidance_scale=7.5, **args) -> None:
        """
        Setup pipeline for generation.

        Args:
            width (int, *optional*):
                image width (default: 768)
            height (int, *optional*):
                image height (default: 768)
            guidance_scale (float, *optional*):
                guidance scale for the model (default: 7.5)
            **args:
                additional arguments passed to BasePipe setup method
        """
        super().setup(**args)
        self.pipe_params.update({
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale
        })

    def gen(self, inputs: dict):
        """
        Generate an image from a prompt.

        Args:
            inputs (dict):
                input dictionary containing the prompt and other parameters

        Returns:
            image (Pil.Image.Image):
                generated image
        """
        kwargs = self.pipe_params.copy()
        # we can override pipe parameters
        # so we update kwargs with inputs after pipe_params
        kwargs.update(inputs)
        # allow for scheduler overwrite
        self.try_set_scheduler(inputs)
        if 'clip_skip' in kwargs and 'StableDiffusionLongPromptWeightingPipeline' in str(type(self.pipe)):
            # not supported
            kwargs.pop('clip_skip')
        image = self.pipe(**kwargs).images[0]
        return image


class Im2ImPipe(BasePipe):
    _autopipeline = AutoPipelineForImage2Image
    _class = StableDiffusionImg2ImgPipeline
    _classxl = StableDiffusionXLImg2ImgPipeline

    def __init__(self, model_id, pipe: Optional[StableDiffusionImg2ImgPipeline] = None, **args):
        super().__init__(model_id=model_id, pipe=pipe, **args)
        self._input_image = None

    def setup(self, fimage, image=None, strength=0.75, gscale=7.5, scale=None, **args):
        """
        Setup pipeline for generation.

        Args:
            fimage (str): File path to the input image.
            image (Image.Image, *optional*): Input image. Defaults to None.
            strength (float, *optional*):
                Strength image modification. Defaults to 0.75. A lower strength values keep result close to the input image.
                 value of 1 means input image more or less ignored.
            gscale (float, *optional*): Guidance scale for the model. Defaults to 7.5.
            scale (list or float, *optional*): Scale factor for the input image. Defaults to None.
            **args: Additional arguments passed to BasePipe setup method.
        """
        super().setup(**args)
        self.fname = fimage
        self._input_image = Image.open(fimage).convert("RGB") if image is None else image
        self._input_image = self.scale_image(self._input_image, scale)
        self.pipe_params.update({
            "strength": strength,
            "guidance_scale": gscale
        })

    def scale_image(self, image, scale):
        """
        Scale the input image.

        Args:
            image (Image.Image): Input image.
            scale (list or float, optional): Scale factor for the input image. Defaults to None.

        Returns:
            Image.Image: Scaled input image.
        """
        if scale is not None:
            if not isinstance(scale, list):
                scale = [8 * (int(image.size[i] * scale) // 8) for i in range(2)]
            return image.resize((scale[0], scale[1]))
        return image

    def get_config(self):
        """
        Return parameters for this pipeline.

        Returns:
            dict: pipeline parameters.
        """
        cfg = super().get_config()
        cfg.update({
            "source_image": self.fname,
        })
        cfg.update(self.pipe_params)
        return cfg

    def gen(self, inputs: dict):
        """
        Generate an image from a prompt and input image.

        Args:
            inputs (dict): Input dictionary containing the prompt and other parameters overriding pipeline configuration.

        Returns:
            Pil.Image.Image: Generated image.
        """
        kwargs = self.pipe_params.copy()
        # we can override pipe parameters
        # so we update kwargs with inputs after pipe_params
        kwargs.update({"image": self._input_image})
        kwargs.update(inputs)
        self.try_set_scheduler(kwargs)
        image = self.pipe(**kwargs).images[0]
        return image


class MaskedIm2ImPipe(Im2ImPipe):
    """
    A pipeline for image-to-image translation with masking.

    MaskedIm2ImPipe is image to image pipeline that uses mask to redraw only certain parts of the input image.
    It can be used as an inpainting pipeline with any non-inpaint models.
    The pipeline computes mask from the difference between
    original image and image with a mask on it. Color of the mask affects the result.
    """

    _class = MaskedStableDiffusionImg2ImgPipeline
    _classxl = MaskedStableDiffusionXLImg2ImgPipeline
    _autopipeline = DiffusionPipeline

    def __init__(self, *args, pipe: Optional[StableDiffusionImg2ImgPipeline] = None, **kwargs):
        """
        Initialize a MaskedIm2ImPipe instance.

        Args:
            *args: arguments passed to Im2ImPipe.
            pipe (StableDiffusionImg2ImgPipeline, *optional*): The underlying pipeline. Defaults to None.
            **kwargs: Additional keyword arguments passed to Im2ImPipe constructor.
        """
        super().__init__(*args, pipe=pipe, **kwargs)
        # convert loaded pipeline if necessary
        if not isinstance(self.pipe, (self._class, self._classxl)):
            self.pipe = self._from_pipe(self.pipe, **kwargs)
        self._mask = None
        self._image_painted = None
        self._original_image = None
        self._mask_blur = None

    def _from_pipe(self, pipe, **args):
        cls = pipe.__class__
        if 'StableDiffusionXLPipeline' in str(cls) :
            return self._classxl(**pipe.components, **args)
        elif 'StableDiffusionPipeline' in str(cls):
            return self._class(**pipe.components, **args)
        raise RuntimeError(f"can't load pipeline from type {cls}")

    def setup(self, original_image=None, image_painted=None, mask=None, blur=4,
              blur_compose=4, sample_mode='sample', scale=None, **kwargs):
        """
        Set up the pipeline.

        Args:
           original_image (str or Image.Image, *optional*):
                The original image. Defaults to None.
           image_painted (str or Image.Image, *optional*):
                The painted image. Defaults to None.
           mask (array-like or Image.Image, *optional*):
               The mask. Defaults to None. If None tt will be computed from the difference
               between original_image and image_painted
           blur (int, *optional*):
                The blur radius for the mask. Defaults to 4.
           blur_compose (int, *optional*):
                The blur radius for composing the original and generated images. Defaults to 4.
           sample_mode (str, *optional*):
                control latents initialisation for the inpaint area, can be one of sample, argmax, random Defaults to 'sample'.
           scale (list or float, *optional*):
                The scale factor for resizing of the input image. Defaults to None.
           **kwargs: Additional keyword arguments passed to Im2ImPipe constructor.
        """
        self._original_image = Image.open(original_image) if isinstance(original_image, str) else original_image
        self._image_painted = Image.open(image_painted) if isinstance(image_painted, str) else image_painted

        input_image = self._image_painted if self._image_painted is not None else self._original_image
        super().setup(fimage=None, image=input_image, scale=scale, **kwargs)
        if self._original_image is not None:
            self._original_image = self.scale_image(self._original_image, scale)
        if self._image_painted is not None:
            self._image_painted = self.scale_image(self._image_painted, scale)
        # there are two options:
        # 1. mask is provided
        # 2. mask is computed from difference between original_image and image_painted
        if image_painted is not None:
            neq = np.any(np.array(self._original_image) != np.array(self._image_painted), axis=-1)
            mask = neq.astype(np.uint8) * 255
        else:
            assert mask is not None
        self._mask = mask

        pil_mask = mask
        if not isinstance(self._mask, Image.Image):
            pil_mask = Image.fromarray(mask)
            if pil_mask.mode != "L":
                pil_mask = pil_mask.convert("L")
        self._mask_blur = self.blur_mask(pil_mask, blur)
        self._mask_compose = self.blur_mask(pil_mask, blur_compose)
        self._sample_mode = sample_mode

    def blur_mask(self, pil_mask, blur):
        mask_blur = pil_mask.filter(ImageFilter.GaussianBlur(radius=blur))
        mask_blur = np.array(mask_blur)
        return np.tile(mask_blur / mask_blur.max(), (3, 1, 1)).transpose(1,2,0)

    def gen(self, inputs):
        inputs = inputs.copy()
        inputs.update(mask=self._mask)
        if 'sample_mode' not in inputs:
            inputs['sample_mode'] = self._sample_mode

        original_image = self._original_image
        original_image = np.array(original_image)
        inputs['original_image'] = original_image / original_image.max()
        img_gen = super().gen(inputs)

        # compose with original using mask
        img_compose = self._mask_compose * img_gen + (1 - self._mask_compose) * self._original_image
        # convert to PIL image
        img_compose = Image.fromarray(img_compose.astype(np.uint8))
        return img_compose


class Cond2ImPipe(BasePipe):
    _class = StableDiffusionControlNetPipeline
    _classxl = StableDiffusionXLControlNetPipeline
    _autopipeline = DiffusionPipeline

    # TODO: set path
    cpath = "./models-cn/"
    cpathxl = "./models-cn-xl/"

    cmodels = {
        "canny": "sd-controlnet-canny",
        "pose": "control_v11p_sd15_openpose",
        "ip2p": "control_v11e_sd15_ip2p",
        "soft-sobel": "control_v11p_sd15_softedge",
        "soft": "control_v11p_sd15_softedge",
        "scribble": "control_v11p_sd15_scribble",
        "depth": "control_v11f1p_sd15_depth",
        "inpaint": "control_v11p_sd15_inpaint",
        "qr": "controlnet_qrcode-control_v1p_sd15"
    }

    cmodelsxl = {
        "soft": "controlnet-softedge-sdxl",
        "scribble": "controlnet-scribble-sdxl",
        "canny": "controlnet-canny-sdxl",
        "pose": "controlnet-openpose-sdxl",
        "depth": "controlnet-depth-sdxl",
        "inpaint": "controlnet-inpaint-sdxl",
        "qr": "controlnet-qr-pattern-sdxl",
    }

    cond_scales_defaults_xl = {
        "pose": 1.0,
        "soft": 0.95, #0.5
        "canny": 0.75,
        "scribble": 0.95,
        "depth": 0.5,
        "inpaint": 1.0,
        "qr": 0.5
    }

    cond_scales_defaults = {
        "canny": 0.75,
        "pose": 1.0,
        "ip2p": 0.5,
        "soft-sobel": 0.3,
        "soft": 0.95, #0.5
        "scribble": 0.95,
        "depth": 0.5,
        "inpaint": 1.0,
        "qr": 1.5
    }

    def __init__(self, model_id, pipe: Optional[StableDiffusionControlNetPipeline] = None,
                 ctypes=["soft"], model_type=None, **args):
        """
        Constructor

        Args:
            model_id (str):
                Path or identifier of the model to load.
            pipe (StableDiffusion(XL)ControlNetPipeline, *optional*):
                An instance of the pipeline to use. If provided, `model_id` won't be used for loading.
            **args:
                Additional arguments passed to the `BasePipe` constructor.

        """
        self.model_type = model_type
        if not isinstance(ctypes, list):
            ctypes = [ctypes]
        self.ctypes = ctypes
        self._condition_image = None
        dtype = torch.float32
        if torch.cuda.is_available():
            dtype = torch.float16
        dtype =  args.get('torch_type', dtype)
        if pipe is None:
            if model_id.endswith('.safetensors'):
                if self.model_type is None:
                    raise RuntimeError(f"model type is not specified for safetensors file {model_id}")
                cpath = self.get_cpath()
                cmodels = self.get_cmodels()
                cnets = [ControlNetModel.from_pretrained(cpath+cmodels[c], torch_dtype=dtype) for c in ctypes]
                super().__init__(model_id=model_id, pipe=pipe, controlnet=cnets, model_type=model_type, **args)
            else:
                cnets = []
                super().__init__(model_id=model_id, pipe=pipe, controlnet=cnets, model_type=model_type, **args)
                # determine model type from pipe
                if isinstance(self.pipe, (self._classxl, StableDiffusionXLPipeline)):
                    t_model_type = ModelType.SDXL
                elif isinstance(self.pipe, (self._class, StableDiffusionPipeline)):
                    t_model_type = ModelType.SD
                else:
                    raise RuntimeError(f"Unexpected model type {type(self.pipe)}")
                self.model_type = t_model_type
                cpath = self.get_cpath()
                cmodels = self.get_cmodels()
                cnets = [ControlNetModel.from_pretrained(cpath+cmodels[c], torch_dtype=dtype).to(self.pipe.device) for c in ctypes]
                if self.model_type == ModelType.SDXL:
                    self.pipe = self._classxl.from_pipe(self.pipe, controlnet=cnets)
                else:
                    self.pipe = self._class.from_pipe(self.pipe, controlnet=cnets)
        else:
            # don't load anything, just reuse pipe
            super().__init__(model_id=model_id, pipe=pipe, **args)
        # FIXME: do we need to setup this specific scheduler here?
        #        should we pass its name in setup to super?
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

    def get_cmodels(self):
        if self.model_type == ModelType.SDXL:
            cmodels = self.cmodelsxl
        elif self.model_type == ModelType.SD:
            cmodels = self.cmodels
        else:
            raise ValueError(f"Unknown controlnet type: {self.model_type}")
        return cmodels

    def get_cpath(self):
        if self.model_type == ModelType.SDXL:
            cpath = self.cpathxl
        elif self.model_type == ModelType.SD:
            cpath = self.cpath
        else:
            raise ValueError(f"Unknown controlnet type: {self.model_type}")
        return cpath

    def setup(self, fimage, width=None, height=None,
              image=None, cscales=None, guess_mode=False, **args):
        """
        Set up the pipeline with the given parameters.

        Args:
            fimage (str):
                The path to the input image file.
            width (int, *optional*):
                The width of the generated image. Defaults to the width of the input image.
            height (int, *optional*):
                The height of the generated image. Defaults to the height of the input image.
            image (PIL.Image.Image, *optional*):
                The input image. Defaults to None. fimage should be None if this argument is provided.
            cscales (list, optional):
                The list of conditioning scales. Defaults to None.
            guess_mode (bool, optional):
                Whether to use guess mode. Defaults to False.
                it enables image generation without text prompt.
            **args: Additional arguments for the pipeline setup.
        """
        super().setup(**args)
        # TODO: allow multiple input images for multiple control nets
        self.fname = fimage
        image = Image.open(fimage) if image is None else image
        self._condition_image = [image]
        if cscales is None:
            cscales = [self.get_default_cond_scales()[c] for c in self.ctypes]
        self.pipe_params.update({
            "width": image.size[0] if width is None else width,
            "height": image.size[1] if height is None else height,
            "controlnet_conditioning_scale": cscales,
            "guess_mode": guess_mode,
            "num_inference_steps": 20
        })

    def get_default_cond_scales(self):
        if self.model_type == ModelType.SDXL:
            cond_scales = self.cond_scales_defaults_xl
        elif self.model_type == ModelType.SD:
            cond_scales = self.cond_scales_defaults
        else:
            raise ValueError(f"Unknown controlnet type: {self.model_type}")
        return cond_scales

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "source_image": self.fname,
            "control_type": self.ctypes
        })
        cfg.update(self.pipe_params)
        return cfg

    def gen(self, inputs):
        """
        Generate an image from the given inputs.

        Args:
            inputs (dict): The dictionary of input parameters.

        Returns:
            PIL.Image.Image: The generated image.
        """
        inputs = {**inputs}
        inputs.update(self.pipe_params)
        inputs.update({"image": self._condition_image})
        image = self.pipe(**inputs).images[0]
        return image


class CIm2ImPipe(Cond2ImPipe):
    """
    A pipeline for conditional image-to-image generation
    where the conditional image is derived from the input image.
    The processing of the input image depends on the specified conditioning type(s).
    """
    def __init__(self, model_id, pipe: Optional[StableDiffusionControlNetPipeline] = None,
                 ctypes=["soft"], model_type=ModelType.SD, **args):

        """
        Initialize the CIm2ImPipe.

        Args:
            model_id (str):
                The identifier of the model to load.
            pipe (StableDiffusion(XL)ControlNetPipeline, *optional*):
                An instance of the pipeline to use. If provided, the model_id won't be used for loading. Defaults to None.
            ctypes (list of str, optional):
                The types of conditioning to apply to the input image. Defaults to ["soft"].
                can be one of canny, pose, soft, soft-sobel, depth, None
            **args:
                Additional arguments passed to the Cond2ImPipe constructor.
        """
        super().__init__(model_id=model_id, pipe=pipe, ctypes=ctypes, model_type=model_type, **args)

        if "soft" in ctypes:
            from controlnet_aux import PidiNetDetector, HEDdetector
            self.processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
            #processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
        if "pose" in ctypes:
            from pytorch_openpose.src.body import Body
            from pytorch_openpose.src import util
            self.body_estimation = Body('pytorch_openpose/model/body_pose_model.pth')
            self.draw_bodypose = util.draw_bodypose
            #hand_estimation = Hand('model/hand_pose_model.pth')
        if "depth" in ctypes:
            from transformers import DPTImageProcessor, DPTForDepthEstimation
            self.dprocessor = DPTImageProcessor.from_pretrained("./models-other/dpt-large")
            self.dmodel = DPTForDepthEstimation.from_pretrained("./models-other/dpt-large")

    def setup(self, fimage, width=None, height=None, image=None, cscales=None, guess_mode=False, **args):
        super().setup(fimage, width, height, image, cscales, guess_mode, **args)
        # Additionally process the input image
        # REM: CIm2ImPipe expects only one image, which can be the base for multiple control images
        self._condition_image = self._proc_cimg(np.asarray(self._condition_image[0]))

    def _proc_cimg(self, oriImg):
        condition_image = []
        for c in self.ctypes:
            if c == "canny":
                image = canny_processor(oriImg)
                condition_image += [Image.fromarray(image)]
            elif c == "pose":
                candidate, subset = self.body_estimation(oriImg)
                canvas = np.zeros(oriImg.shape, dtype = np.uint8)
                canvas = self.draw_bodypose(canvas, candidate, subset)
                #canvas[:, :, [2, 1, 0]]
                condition_image += [Image.fromarray(canvas)]
            elif c == "soft":
                condition_image += [self.processor(oriImg)]
            elif c == "soft-sobel":
                edge = sobel_processor(oriImg)
                condition_image += [Image.fromarray(edge)]
            elif c == "depth":
                image = Image.fromarray(oriImg)
                inputs = self.dprocessor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.dmodel(**inputs)
                    predicted_depth = outputs.predicted_depth
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )
                output = prediction.squeeze().cpu().numpy()
                formatted = (output * 255 / np.max(output)).astype("uint8")
                condition_image += [Image.fromarray(formatted)]
            else:
                condition_image += [Image.fromarray(oriImg)]
        return condition_image


# TODO: does it make sense to inherint it from Cond2Im or CIm2Im ?
class InpaintingPipe(BasePipe):
    """
    A pipeline for inpainting images using ControlNet models.
    """

    _class = StableDiffusionControlNetInpaintPipeline
    _classxl = StableDiffusionXLControlNetInpaintPipeline
    _autopipeline = AutoPipelineForInpainting

    def __init__(self, model_id, pipe: Optional[StableDiffusionControlNetPipeline] = None,
                 **args):
        """
        Initialize the InpaintingPipe.

        Args:
            model_id (str):
                The identifier of the model to load.
            pipe (StableDiffusion(XL)ControlNetPipeline, *optional*):
                An instance of the pipeline to use. If provided, the model_id won't be used for loading. Defaults to None.
            **args:
                Additional arguments passed to the BasePipe constructor.
        """
        dtype = torch.float32
        if torch.cuda.is_available():
            dtype = torch.float16
        dtype =  args.get('torch_type', dtype)
        cnet = ControlNetModel.from_pretrained(
            Cond2ImPipe.cpath+Cond2ImPipe.cmodels["inpaint"], torch_dtype=dtype)
        super().__init__(model_id=model_id, pipe=pipe, controlnet=cnet, **args)
        # FIXME: do we need to setup this specific scheduler here?
        #        should we pass its name in setup to super?
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def setup(self, fimage, mask_image, image=None, **args):
        """
        Set up the pipeline for inpainting with the given image and mask.

        Args:
            fimage:
                The path to the base image to be inpainted.
            mask_image:
                The mask image indicating the areas to be inpainted.
            image (optional):
                An additional image input for processing. Defaults to None.
            **args:
                Additional arguments passed to the BasePipe setup method.
        """
        super().setup(**args)
        # TODO: allow multiple input images for multiple control nets
        self.fname = fimage
        self._init_image = Image.open(fimage) if image is None else image
        self._mask_image = mask_image
        self._control_image = self._make_inpaint_condition(self._init_image, mask_image)

        self.pipe_params.update({
            # TODO: check if condtitioning_scale and guess_mode are in this pipeline and what is their effect
            # "controlnet_conditioning_scale": cscales,
            # "guess_mode": guess_mode,
            "eta": 0.1, # FIXME: is it needed?
            "num_inference_steps": 20
        })

    def gen(self, inputs):
        """
        Generate an inpainted image using the pipeline.

        Args:
            inputs (dict):
                A dictionary of additional parameters to pass to the pipeline.

        Returns:
            PIL.Image:
                The generated inpainted image.
        """

        inputs = {**inputs}
        inputs.update(self.pipe_params)
        inputs.update({
            "image": self._init_image,
            "mask_image": self._mask_image,
            "control_image": self._control_image
        })
        image = self.pipe(**inputs).images[0]
        return image

    def _make_inpaint_condition(self, image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image


def canny_processor(oriImg):
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(oriImg, low_threshold, high_threshold)
    image = image[:, :, None]
    return np.concatenate([image, image, image], axis=2)

def sobel_processor(oriImg, ksize=9):
    gray = cv2.cvtColor(oriImg, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray, 0, 1, 0, ksize=ksize, scale=1)
    y = cv2.Sobel(gray, 0, 0, 1, ksize=ksize, scale=1)
    absx= cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    edge = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    #edge = np.hypot(x, y)
    return edge
