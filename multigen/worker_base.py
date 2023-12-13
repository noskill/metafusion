from datetime import datetime
import threading
from collections import deque
import random
import yaml
from pathlib import Path
import logging

class ServiceThreadBase(threading.Thread):
    def __init__(self, cfg_file):
        super().__init__(name='imgen', daemon=False)
        cfg_file = Path(cfg_file)
        self.cwd = cfg_file.parent
        with open(cfg_file, "r") as f:
            self.config = yaml.safe_load(f)
        with open(self.cwd/self.config["model_list"], "r") as f:
            self.models = yaml.safe_load(f)
        if 'logging_folder' in self.config:
            logname = self.cwd / (self.config['logging_folder'] + datetime.today().strftime('%Y-%m-%d') + ".log")
            logging.basicConfig(filename=logname)
        self.logger = logging.getLogger(__name__)
        self.sessions = {}
        self.queue = deque()
        self.qlimit = 5
        self.max_img_cnt = 20
        self._lock = threading.Lock()
        self._stop = False

    def open_session(self, **args):
        user = args["user"]
        with self._lock:
            for s in self.sessions:
                if self.sessions[s]["user"] == user:
                    return { "error": f"User {user} already has an open session",
                             "session_id": s
                           }
            if args["model"] not in self.models['base']:
                raise RuntimeError(f"Unknown model {args['model']}")
            id = str(random.randint(0, 1024*1024*1024*4-1))
            self.sessions[id] = {**args}
            self.sessions[id]["images"] = []
            self.logger.info("OPENING session %s", id)
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
            if len(self.queue) >= self.qlimit:
                return { "error": "Server is busy" }
            for q in self.queue:
                if q["session_id"] == args["session_id"]:
                    return { "error": "The job for this session already exists" }
            a = {**args}
            a["count"] = int(a["count"])
            if a["count"] <= 0:
                return { "warning": f"no images to generate ({a['count']}), job has not been created" }
            r = { "status": "success" }
            if a["count"] > self.max_img_cnt:
                a["count"] = self.max_img_cnt
                r["warning"] = f"maximum image count exceeded {self.max_img_cnt}"
            self.queue.appendleft(a)
            r["queue_number"] = len(self.queue)-1
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
        result["status"] = "open"
        for i in range(len(self.queue)):
            q = self.queue[i]
            if q["session_id"] == session_id:
                result["status"] = "running" if i == 0 else "queued"
                result["next_job"] = {
                    "queue_number": i,
                    "image_count": q["count"]
                }
                break
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