import os
import sys

import imageio
import numpy as np

from pathlib import Path
from datetime import datetime



def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    print(dir_path)
    try:
        dir_path.mkdir(exist_ok=True, parents=True)
    except OSError:
        pass
    return dir_path


class VideoRecorder(object):
    def __init__(self, name='test',height=256, width=256, fps=50):
        current_date = datetime.now()
        date_string = current_date.strftime('%Y-%m-%d')
        storage_path = Path(".")
        video_path = storage_path / 'video' 
        root_dir = video_path
        self.save_dir = root_dir/ name
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []


    def record_original(self, env):
        if self.enabled:
            frame = env.physics.render(height=self.height,
                                       width=self.width,
                                       camera_id=0)
            print(frame[:1,0,0])   #(256, 256, 3)
            self.frames.append(frame)

    def record(self, frame):
       
        self.frames.append(frame)

    def save(self, file_name):
        #path = os.path.join(self.save_dir, file_name)
        path = self.save_dir / file_name
        imageio.mimsave(path, self.frames, fps=self.fps)
