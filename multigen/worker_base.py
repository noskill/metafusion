import os
from datetime import datetime
from typing import Type
import threading
from queue import Queue, Full
import random
import yaml
from pathlib import Path
import logging
from .pipes import Prompt2ImPipe, MaskedIm2ImPipe, Cond2ImPipe, Im2ImPipe
from .loader import Loader


class ServiceThreadBase(threading.Thread):
    """
    Base class implementing the worker thread for image generation
    """
    def __init__(self, cfg_file):
        super().__init__(name='imgen', daemon=False)
        cfg_file = Path(cfg_file)
        self.queue_num = 0
        self.cwd = cfg_file.parent
        with open(cfg_file, "r") as f:
            self.config = yaml.safe_load(f)
        with open(self.cwd/self.config["model_list"], "r") as f:
            self.models = yaml.safe_load(f)
        if 'logging_folder' in self.config:
            logname = self.cwd / (self.config['logging_folder'] + datetime.today().strftime('%Y-%m-%d') + ".log")
            if not os.path.exists(self.config['logging_folder']):
                os.mkdir(self.config['logging_folder'])
            logging.basicConfig(filename=logname, level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.sessions = {}
        self.qlimit = 5
        self.queue = Queue(self.qlimit)
        self.max_img_cnt = 20
        self._lock = threading.RLock()
        self._stop = False
        self._loader = Loader()
        self._pipe_name_to_pipe = dict()
        for p in self.models['pipes']:
            class_name = self.models['pipes'][p]['classname']
            self._pipe_name_to_pipe[p] = globals()[class_name]
        self.logger.debug('pipe name to pipe %s', str(self._pipe_name_to_pipe))
        self.logger.info('service is running')

    @property
    def pipes(self):
        """
        Get the list of available pipes.
        """
        return self.models['pipes']

    def open_session(self, **args):
        """
        Open a new session for a user.

        Parameters:
           **args (dict): Keyword arguments containing the session details.
               user (str): The username of the session owner.
               model (str): The name of the model to use for the session, the one specified in config
                   file under "base" field.
               pipe: (str): The name of the pipe to use for the session, the one specified in models config
                   file under "pipes" field.
               project (str): The name of the project to generate images.

        Returns:
           dict: A dictionary containing the session ID(session_id) and an error message if applicable.
        """
        user = args["user"]
        with self._lock:
            for s in self.sessions:
                if self.sessions[s]["user"] == user:
                    return { "error": f"User {user} already has an open session",
                             "session_id": s
                           }
            if args["model"] not in self.models['base']:
                raise RuntimeError(f"Unknown model {args['model']}")
            for lora in args.get('loras', []):
                if lora not in self.models['lora']:
                    raise RuntimeError(f"Unknown lora model {lora}")
            id = str(random.randint(0, 1024*1024*1024*4-1))
            self.sessions[id] = {**args}
            self.sessions[id]["images"] = []
            self.logger.info("OPENING session %s, model %s", id, args['model'])
            return { "session_id": id }

    def close_session(self, session_id):
        with self._lock:
            if session_id not in self.sessions:
                raise RuntimeError(f"Session {session_id} is not open")
            del self.sessions[session_id]
        return { "status": "success" }

    def queue_gen(self, **args):
        self.logger.info("REQUESTED FOR QUEUE: " + str(args))
        with self._lock:
            if args["session_id"] not in self.sessions:
                return { "error": "Session is not open" }
           # for q in self.queue:
           #     if q["session_id"] == args["session_id"]:
           #         return { "error": "The job for this session already exists" }
            a = {**args}
            a["count"] = int(a["count"])
            if a["count"] <= 0:
                return { "warning": f"no images to generate ({a['count']}), job has not been created" }
            r = { "status": "success" }
            if a["count"] > self.max_img_cnt:
                a["count"] = self.max_img_cnt
                r["warning"] = f"maximum image count exceeded {self.max_img_cnt}"
            try:
                self.queue.put(a)
            except Full:
                return { "error": "Server is busy" }

            r["queue_number"] = self.queue_num
            self.queue_num += 1
            self.logger.info(f'queued a job {a}')
            return r

    def get_image_count(self, session_id):
        with self._lock:
            if session_id not in self.sessions:
                raise RuntimeError(f"Session {session_id} does not exist")
            return {
                "image_number": len(self.sessions[session_id]["images"]),
                "comment": f"already generated number of images in session {session_id}"
            }

    def get_session_info(self, session_id):
        if session_id not in self.sessions:
            return { "status": f"session {session_id} does not exist" }
        sess = self.sessions[session_id]
        result = {}
        result["available_images"] = len(sess["images"])
        result["next_job"] = "None"
        result["status"] = sess.get('status', 'open') 
        return result

    def get_image_pathname(self, session_id, img_idx):
        with self._lock:
            if session_id not in self.sessions:
                raise RuntimeError(f"Session {session_id} does not exist")
            sess = self.sessions[session_id]
            imgs = sess["images"]
            if img_idx is not None and (img_idx < 0 or img_idx >= len(imgs)):
                r = self.get_session_info(session_id)
                r.update({ "warning": "Incorrect image index" })
                return r
            # TODO: root dir
            path = self.cwd/"_projects"/sess["user"]/sess["project"]
            if img_idx is None:
                return path
            # TODO: image file names are saved with path now... getting path and name seem to be independent
            return imgs[img_idx]

    def stop(self):
        self._stop = True

    def run(self):
        raise NotImplementedError('Define run in %s' % (self.__class__.__name__))

    def get_pipe_class(self, name: str) -> Type['BasePipe']:
        return self._pipe_name_to_pipe[name]

