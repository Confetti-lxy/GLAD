import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import random


class Analyzer:
    """ Wraps the analyzer for dataset. """
    def __init__(self, name: str, dataset_name: str):
        self.name = name
        self.dataset_name = dataset_name

        env = env_settings()
        self.results_dir = '{}/{}'.format(env.results_path, self.name)

        analyzer_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(analyzer_module_abspath):
            analyzer_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.analyzer_class = analyzer_module.get_analyzer_class()
        else:
            self.analyzer_class = None

    def create_analyzer(self):
        analyzer = self.analyzer_class(self.dataset_name)
        return analyzer

    def run_sequence_all(self, seq):
        """Run analyzer on sequence.
        args:
            seq: Sequence to run the analyzer on.
        """
        # Get init information
        init_info = seq.init_info()

        analyzer = self.create_analyzer()

        output = self._analyze_sequence_all(analyzer, seq, init_info)
        return output
    
    def _analyze_sequence_all(self, analyzer, seq, init_info):
        output = {'logits': []}

        # Initialize
        image = self._read_image(seq.frames[0])
        logits = analyzer.initialize_all(image, init_info)
        logits = logits.cpu().detach().numpy()
        # Record
        output['logits'].append(logits[0, 0])

        # Track, select all frames
        for frame_num, frame_path in enumerate(seq.frames[20::20]):
            image = self._read_image(frame_path)
            logits = analyzer.track_all(image)
            logits = logits.cpu().detach().numpy()
            # Record
            output['logits'].append(logits[0, 0])

        return output

    def run_sequence_once(self, seq):
        """Run analyzer on sequence.
        args:
            seq: Sequence to run the analyzer on.
        """
        # Get init information
        init_info = seq.init_info()

        analyzer = self.create_analyzer()

        output = self._analyze_sequence_once(analyzer, seq, init_info)
        return output

    def _analyze_sequence_once(self, analyzer, seq, init_info):
        output = {'logits': [], 'probs': []}

        # Initialize
        image = self._read_image(seq.frames[0])
        analyzer.initialize_once(image, init_info)

        # Track, select a random frame near the middle(40%~60%) of the video
        begin_frame = int(len(seq.frames) * 0.4) + 1
        end_frame = int(len(seq.frames) * 0.6) - 1
        frame_num = random.randint(begin_frame, end_frame)
        frame_path = seq.frames[frame_num]
        image = self._read_image(frame_path)
        info = seq.frame_info(frame_num)
        logits_per_text, probs = analyzer.track_once(image, info)

        logits_per_text = logits_per_text.cpu().detach().numpy()
        probs = probs.cpu().detach().numpy()

        # Record
        output['logits'].append(logits_per_text[0, 0])
        output['logits'].append(logits_per_text[0, 1])
        output['probs'].append(probs[0, 0])
        output['probs'].append(probs[0, 1])

        return output

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")
