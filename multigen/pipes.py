import importlib
from enum import Enum

import torch

from PIL import Image, ImageFilter
import cv2
import copy
import numpy as np
from typing import Optional, Type
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline
from diffusers import StableDiffusionControlNetInpaintPipeline, StableDiffusionXLControlNetInpaintPipeline, DDIMScheduler
from diffusers.schedulers import KarrasDiffusionSchedulers

from .pipelines.masked_stable_diffusion_img2img import MaskedStableDiffusionImg2ImgPipeline
from transformers import CLIPProcessor, CLIPTextModel
#from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
# from diffusers import StableDiffusionKDiffusionPipeline


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
    _class = None

    def __init__(self, model_id: str,
                 sd_pipe_class: Optional[Type[DiffusionPipeline]]=None,
                 pipe: Optional[DiffusionPipeline] = None, **args):
        self.pipe = pipe
        self._scheduler = None
        self._hypernets = []
        self._model_id = model_id
        self.pipe_params = dict()
        # Creating a stable diffusion pipeine
        args = {**args}
        if 'torch_dtype' not in args:
            args['torch_dtype']=torch.float16

        if self.pipe is None:
            constructor_args = dict()
            if isinstance(self, Cond2ImPipe):
                constructor_args['controlnet'] = args['controlnet']

            if sd_pipe_class is None:
                if model_id.endswith('.safetensors'):
                    try:
                        self.pipe = StableDiffusionPipeline.from_single_file(model_id, **args)
                    except TypeError as e:
                        self.pipe = StableDiffusionXLPipeline.from_single_file(model_id, **args)
                else:
                    # we can't use specific class, because we dont know if it is sdxl
                    self.pipe = DiffusionPipeline.from_pretrained(model_id, **args)
                    if 'custom_pipeline' not in args:
                        # create correct class if custom_pipeline is not specified
                        # at this stage we know that the model is sdxl or sd
                        self.pipe = self.from_pipe(self.pipe, **constructor_args)

            else:
                if model_id.endswith('.safetensors'):
                    self.pipe = sd_pipe_class.from_single_file(model_id, **args)
                else:
                    self.pipe = sd_pipe_class.from_pretrained(model_id, **args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipe.to(device)
        # self.pipe.enable_attention_slicing()
        # self.pipe.enable_vae_slicing()
        self.pipe.vae.enable_tiling()
        # --- the best one and seems to be enough ---
        # self.pipe.enable_sequential_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention() # attention_op=MemoryEfficientAttentionFlashAttentionOp)
        # self.pipe.vae.enable_xformers_memory_efficient_attention() # attention_op=None)

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def hypernets(self):
        return self._hypernets

    @property
    def model_id(self):
        return self._model_id

    def try_set_scheduler(self, inputs):
        # allow for scheduler overwrite
        scheduler = inputs.get('scheduler', None)
        if scheduler is not None and self.pipe is not None:
            sch_set = add_scheduler(self.pipe, scheduler=scheduler)
            if sch_set:
                self._scheduler = sch_set
            inputs.pop('scheduler')

    def add_hypernet(self, path, multiplier=None):
        from . hypernet import add_hypernet, clear_hypernets, Hypernetwork
        hypernetwork = Hypernetwork()
        hypernetwork.load(path)
        self._hypernets.append(path)
        hypernetwork.set_multiplier(multiplier if multiplier else 1.0)
        hypernetwork.to(self.pipe.unet.device)
        hypernetwork.to(self.pipe.unet.dtype)
        add_hypernet(self.pipe.unet, hypernetwork)

    def clear_hypernets(self):
        clear_hypernets(self.pipe.unet)
        self._hypernets = []

    def get_config(self):
        cfg = {"hypernetworks": self.hypernets }
        cfg.update({"model_id": self.model_id })
        cfg.update(self.pipe_params)
        return cfg

    def setup(self, steps=50, **args):
        self.pipe_params = { 'num_inference_steps': steps }
        if 'clip_skip' in args:
            # TODO? add clip_skip to config?
            clip_skip = args['clip_skip']
            assert clip_skip >= 0
            assert clip_skip <= 10
            if clip_skip:
                prev_encoder = self.pipe.text_encoder
                prev_config = prev_encoder.config
                # if we need less or equal number of hidden layers
                if 12 - clip_skip <= prev_config.num_hidden_layers:
                    config = copy.copy(prev_config)
                    config.num_hidden_layers = 12 - clip_skip
                    self.pipe.text_encoder = CLIPTextModel(config)
                    self.pipe.text_encoder.load_state_dict(prev_encoder.state_dict(), strict=False)
                else:  # we need more hidden layers
                    self.pipe.text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder",
                                                                           num_hidden_layers=12 - clip_skip)
                self.pipe.text_encoder.to(prev_encoder.device)
                self.pipe.text_encoder.to(prev_encoder.dtype)
        if 'scheduler' in args:
            # TODO? add scheduler to config?
            self.try_set_scheduler(dict(scheduler=args['scheduler']))

    def from_pipe(self, pipe, **args):
        if isinstance(pipe, StableDiffusionXLPipeline):
            return self._classxl(**pipe.components, **args)
        if isinstance(pipe, StableDiffusionPipeline):
            return self._class(**pipe.components, **args)
        # it's a custom pipeline
        return pipe


class Prompt2ImPipe(BasePipe):
    _class = StableDiffusionPipeline
    _classxl = StableDiffusionXLPipeline

    def __init__(self, model_id: str,
                 pipe: Optional[StableDiffusionPipeline] = None,
                 lpw=False, **args):
        if not lpw:
            super().__init__(model_id=model_id, pipe=pipe, **args)
        else:
            #StableDiffusionKDiffusionPipeline
            super().__init__(model_id=model_id, pipe=pipe, custom_pipeline="lpw_stable_diffusion", **args)

    def setup(self, width=768, height=768, guidance_scale=7.5, **args):
        super().setup(**args)
        self.pipe_params.update({
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale
        })

    def gen(self, inputs):
        inputs = {**inputs}
        inputs.update(self.pipe_params)
        # allow for scheduler overwrite
        self.try_set_scheduler(inputs)
        image = self.pipe(**inputs).images[0]
        return image


class Im2ImPipe(BasePipe):

    _class = StableDiffusionImg2ImgPipeline
    _classxl = StableDiffusionXLImg2ImgPipeline

    def __init__(self, model_id, pipe: Optional[StableDiffusionImg2ImgPipeline] = None, **args):
        super().__init__(model_id=model_id, pipe=pipe, **args)
        self._input_image = None

    def setup(self, fimage, image=None, strength=0.75, gscale=7.5, scale=None, **args):
        super().setup(**args)
        self.fname = fimage
        self._input_image = Image.open(fimage).convert("RGB") if image is None else image
        if scale is not None:
            if not isinstance(scale, list):
                scale = [8 * (int(self._input_image.size[i] * scale) // 8) for i in range(2)]
            self._input_image = self._input_image.resize((scale[0], scale[1]))
        self.pipe_params.update({
            "strength": strength,
            "guidance_scale": gscale
        })

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "source_image": self.fname,
        })
        cfg.update(self.pipe_params)
        return cfg

    def gen(self, inputs):
        inputs = {**inputs}
        inputs.update(self.pipe_params)
        inputs.update({"image": self._input_image})
        self.try_set_scheduler(inputs)
        image = self.pipe(**inputs).images[0]
        return image


class MaskedIm2ImPipe(Im2ImPipe):
    _class = MaskedStableDiffusionImg2ImgPipeline

    def __init__(self, *args, pipe: Optional[StableDiffusionImg2ImgPipeline] = None, **kwargs):
        super().__init__(*args, pipe=pipe, **kwargs)
        self._mask = None
        self._image_painted = None
        self._original_image = None
        self._mask_blur = None

    def setup(self, original_image=None, image_painted=None, mask=None, blur=4, **kwargs):
        self._original_image = Image.open(original_image) if isinstance(original_image, str) else original_image
        self._image_painted = Image.open(image_painted) if isinstance(image_painted, str) else image_painted
        # there are two options:
        # 1. mask is provided
        # 2. mask is computed from difference between original_image and image_painted
        if image_painted is not None:
            neq = np.any(np.array(self._original_image) != np.array(self._image_painted), axis=-1)
            mask = neq.astype(np.uint8) * 255
        else:
            assert mask is not None
        self._mask = mask
        input_image = self._image_painted if self._image_painted is not None else self._original_image
        input_image = np.array(input_image)
        super().setup(fimage=None, image=input_image / input_image.max(), **kwargs)
        pil_mask = mask
        if not isinstance(self._mask, Image.Image):
            pil_mask = Image.fromarray(mask)
            if pil_mask.mode != "L":
                pil_mask = pil_mask.convert("L")
        mask_blur = pil_mask.filter(ImageFilter.GaussianBlur(radius=blur))
        mask_blur = np.array(mask_blur)
        self._mask_blur = np.tile(mask_blur / mask_blur.max(), (3, 1, 1)).transpose(1,2,0)

    def gen(self, inputs):
        inputs = inputs.copy()
        inputs.update(mask=self._mask)
        img_gen = super().gen(inputs)

        # compose with original using mask
        img_compose = self._mask_blur * img_gen + (1 - self._mask_blur) * self._original_image
        # convert to PIL image
        img_compose = Image.fromarray(img_compose.astype(np.uint8))
        return img_compose


class ControlnetType(Enum):
    SD = 1
    SDXL = 2


class Cond2ImPipe(BasePipe):
    _class = StableDiffusionControlNetPipeline
    _classxl = StableDiffusionXLControlNetPipeline

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
        "qr": "controlnet-qr-pattern-sdxl",
    }

    cond_scales_defaults_xl = {
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
                 ctypes=["soft"], model_type=ControlnetType.SD, **args):
        self.model_type = model_type
        if not isinstance(ctypes, list):
            ctypes = [ctypes]
        self.ctypes = ctypes
        self._condition_image = None
        dtype = torch.float16 if 'torch_type' not in args else args['torch_type']
        cpath = self.get_cpath()
        cmodels = self.get_cmodels()
        sd_class = self.get_sd_class()
        cnets = None
        if pipe is None:
            cnets = [ControlNetModel.from_pretrained(cpath+cmodels[c], torch_dtype=dtype) for c in ctypes]
        super().__init__(model_id=model_id, pipe=pipe, sd_pipe_class=sd_class, controlnet=cnets, **args)
        # FIXME: do we need to setup this specific scheduler here?
        #        should we pass its name in setup to super?
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

    def get_cmodels(self):
        if self.model_type == ControlnetType.SDXL:
            cmodels = self.cmodelsxl
        elif self.model_type == ControlnetType.SD:
            cmodels = self.cmodels
        else:
            raise ValueError(f"Unknown controlnet type: {self.model_type}")
        return cmodels

    def get_cpath(self):
        if self.model_type == ControlnetType.SDXL:
            cpath = self.cpathxl
        elif self.model_type == ControlnetType.SD:
            cpath = self.cpath
        else:
            raise ValueError(f"Unknown controlnet type: {self.model_type}")
        return cpath

    def get_sd_class(self):
        if self.model_type == ControlnetType.SDXL:
            cclass = self._classxl
        elif self.model_type == ControlnetType.SD:
            cclass = self._class
        else:
            raise ValueError(f"Unknown controlnet type: {self.model_type}")
        return cclass

    def setup(self, fimage, width=None, height=None, image=None, cscales=None, guess_mode=False, **args):
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
        if self.model_type == ControlnetType.SDXL:
            cond_scales = self.cond_scales_defaults_xl
        elif self.model_type == ControlnetType.SD:
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
        inputs = {**inputs}
        inputs.update(self.pipe_params)
        inputs.update({"image": self._condition_image})
        image = self.pipe(**inputs).images[0]
        return image


class CIm2ImPipe(Cond2ImPipe):

    def __init__(self, model_id, pipe: Optional[StableDiffusionControlNetPipeline] = None,
                 ctypes=["soft"], model_type=ControlnetType.SD, **args):
        super().__init__(model_id=model_id, pipe=pipe, ctypes=ctypes, model_type=model_type, **args)
        # The difference from Cond2ImPipe is that the conditional image is not
        # taken as input but is obtained from an ordinary image, so this image
        # should be processed, and the processor depends on the conditioning type
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
    _class = StableDiffusionControlNetInpaintPipeline
    _classxl = StableDiffusionXLControlNetInpaintPipeline

    def __init__(self, model_id, pipe: Optional[StableDiffusionControlNetPipeline] = None,
                 **args):
        dtype = torch.float16 if 'torch_type' not in args else args['torch_type']
        cnet = ControlNetModel.from_pretrained(
            Cond2ImPipe.cpath+Cond2ImPipe.cmodels["inpaint"], torch_dtype=dtype)
        super().__init__(model_id=model_id, pipe=pipe, controlnet=cnet, **args)
        # FIXME: do we need to setup this specific scheduler here?
        #        should we pass its name in setup to super?
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def setup(self, fimage, mask_image, image=None, **args):
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
