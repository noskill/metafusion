from typing import Type, List
import copy
from contextlib import nullcontext
import torch
import logging
import threading
from diffusers import DiffusionPipeline, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline
from diffusers.utils import is_accelerate_available
if is_accelerate_available():
    from accelerate import init_empty_weights
else:
    init_empty_weights = nullcontext


logger = logging.getLogger(__file__)

def copy_pipe(pipe):
    copy = pipe.__class__(**pipe.components)
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        copy.unet = copy.unet.__class__.from_config(copy.unet.config)
        copy.text_encoder = copy.text_encoder.__class__(pipe.text_encoder.config)
        if hasattr(copy, 'text_encoder_2'):
            copy.text_encoder_2 = copy.text_encoder_2.__class__(pipe.text_encoder_2.config)
    # assign=True is needed since our copy is on "meta" device, i.g. weights are empty
    copy.unet.load_state_dict(pipe.unet.state_dict(), assign=True)
    copy.text_encoder.load_state_dict(pipe.text_encoder.state_dict(), assign=True)
    if hasattr(copy, 'text_encoder_2'):
        copy.text_encoder_2.load_state_dict(pipe.text_encoder_2.state_dict(), assign=True)
    return copy


class Loader:
    """
    class for loading diffusion pipelines from files.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._cpu_pipes = dict()
        # idx -> list of (model_id, pipe) pairs
        self._gpu_pipes = dict()

    def get_gpu(self, model_id) -> List[int]:
        """
        return list of gpus with loaded model
        """
        with self._lock:
            result = list()
            for idx, items in self._gpu_pipes.items():
                for (model, _) in items:
                    if model == model_id:
                        result.append(idx)
            return result

    def load_pipeline(self, cls: Type[DiffusionPipeline], path, torch_dtype=torch.float16, device=None,
            **additional_args):
        with self._lock:
            logger.debug(f'looking for pipeline {cls} from {path} on {device}')
            result = None
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cuda':
                idx = device.index 
                gpu_pipes = self._gpu_pipes.get(idx, [])
                for (key, value) in gpu_pipes:
                    if key == path:
                        logger.debug(f'found pipe in gpu cache {key}')
                        result = self.from_pipe(cls, value, additional_args)
                        logger.debug(f'created pipe from gpu cache {key} on {device}')
                        return result 
            for (key, pipe) in self._cpu_pipes.items():
                if key == path:
                    logger.debug(f'found pipe in cpu cache {key}')
                    result = self.from_pipe(cls, copy.deepcopy(pipe), additional_args)
                    break
            if result is None:
                logger.info(f'not found {path} in cache, loading')
                if path.endswith('safetensors'):
                    result = cls.from_single_file(path, **additional_args)
                else:
                    result = cls.from_pretrained(path, **additional_args)
            if device.type == 'cuda':
                self.clear_cache(device)
            result = result.to(dtype=torch_dtype, device=device)
            self.cache_pipeline(result, path)
            result = copy_pipe(result)
            assert result.device == device
            logger.debug(f'returning {type(result)} from {path} on {result.device}')
            return result
    
    def from_pipe(self, cls, pipe, additional_args):
        pipe = copy_pipe(pipe)
        components = pipe.components
        if issubclass(cls, StableDiffusionXLControlNetPipeline) or issubclass(cls, StableDiffusionControlNetPipeline):
            # todo: keep controlnets in cache explicitly
            if 'controlnet' in additional_args:
                components.pop('controlnet')
            return cls(**components, **additional_args)
        # handling the case when the model in cache has controlnet in it
        # but we don't need it
        if 'controlnet' in components:
            components.pop('controlnet')
        return cls(**components, **additional_args)

    def cache_pipeline(self, pipe: DiffusionPipeline, model_id):
        with self._lock:
            device = pipe.device
            if model_id not in self._cpu_pipes:
            # deepcopy is needed since Module.to is an inplace operation
                self._cpu_pipes[model_id] = copy.deepcopy(pipe.to('cpu'))
            pipe.to(device)
            if pipe.device.type == 'cuda':
                self._store_gpu_pipe(pipe, model_id)
            logger.debug(f'storing {model_id} on {pipe.device}')

    def clear_cache(self, device):
        logger.debug(f'clear_cache pipelines from {device}')
        with self._lock:
            if device.type == 'cuda':
                self._gpu_pipes[device.index] = [] 

    def _store_gpu_pipe(self, pipe, model_id):
        idx = pipe.device.index 
        # for now just clear all other pipelines
        self._gpu_pipes[idx] = [(model_id, pipe)]

    def remove_pipeline(self, model_id):
        self._cpu_pipes.pop(model_id)

    def get_pipeline(self, model_id, device=None):
        with self._lock:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cuda':
                idx = device.index
                gpu_pipes = self._gpu_pipes.get(idx, ())
                for (key, value) in gpu_pipes:
                    if key == model_id:
                        return value
            for (key, pipe) in self._cpu_pipes.items():
                if key == model_id:
                    return pipe

            return None
