import os
import shutil
import time
import torch
import unittest
import yaml
import logging
from multigen.prompting import Cfgen
from multigen.sessions import GenSession
from multigen.pipes import Prompt2ImPipe

from multigen.worker import ServiceThread
from multigen.log import setup_logger

setup_logger()



nprompt = "jpeg artifacts, blur, distortion, watermark, signature, extra fingers, fewer fingers, lowres, bad hands, duplicate heads, bad anatomy, bad crop"

prompt = "Close-up portrait of a woman wearing suit posing with black background, rim lighting, octane, unreal"
seed = 383947828373273


class WorkerTestCase(unittest.TestCase):
    def setUp(self):
        self.cfg_file = 'config.yaml'
        self.models_conf = yaml.safe_load(open('models.yaml'))
        self.worker = ServiceThread(self.cfg_file)
        self.worker.start()

    def test_multisessions(self):
        if not torch.cuda.is_available():
            print('no gpu to run test_multisessions')
            return True

        pipe = "prompt2image" 
        num_models = 4
        sessions = []
        for model in self.models_conf['base'].keys():
            result = self.worker.open_session(
                user='test' + model,
                project="results" + model,
                model=model,
                pipe=pipe,
            )
            if 'error' in result:
                raise RuntimeError("can't open session")
            sessions.append(result['session_id'])
        
        count = 5 
        c = 0
        def on_new_image(*args, **kwargs):
            print(args, kwargs)
            nonlocal c
            c += 1
        
        num_runs = 15
        for i in range(num_runs):
            if len(sessions) - 1 < i:
                i %= len(sessions)
            sess_id = sessions[i]
            self.worker.queue_gen(session_id=sess_id, 
        					images=None,
        					prompt=prompt, pipe='Prompt2ImPipe',
                            image_callback=on_new_image,
        				    lpw=True,
        					width=1024, height=1024, steps=5, 
        					timestep_spacing='trailing', 
        					guidance_scale=0, 
        					scheduler='EulerAncestralDiscreteScheduler',
                            count=count,
                            seeds=[seed + i for i in range(count)],
                            )
        timeout = 1000        
        while timeout and count * num_runs != c:
            time.sleep(1)
            timeout -= 1
        self.assertEqual(count * num_runs, c, 'not enough images generated') 
        
    def tearDown(self):
        self.worker.stop()
        for dir in os.listdir('./_projects'):
            if dir.startswith('test'):
                shutil.rmtree(os.path.join('_projects', dir))


if __name__ == '__main__':
    unittest.main()
