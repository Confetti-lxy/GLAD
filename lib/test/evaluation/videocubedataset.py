import os

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import json
import pandas as pd


class VideoCubeDataset(BaseDataset):
    """
    VideoCube test set
    """

    def __init__(self, version="tiny"):
        super().__init__()

        # self.split = split
        self.split = "test"
        self.version = version

        f = open(
            os.path.join(
                os.path.split(os.path.realpath(__file__))[0], "videocube.json"
            ),
            "r",
            encoding="utf-8",
        )
        self.infos = json.load(f)[self.version]
        f.close()

        self.sequence_list = self.infos[self.split]

        print("sequence_list", self.sequence_list)

        self.base_path = self.env_settings.videocube_path


        # if split == "test" or split == "val":
        #     self.base_path = self.env_settings.videocube_path  #
        # else:
        #     self.base_path = self.env_settings.videocube_path

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = "{}/{}/{}/{}.txt".format(
            self.base_path, "attribute", "groundtruth", sequence_name
        )

        # /home/luoxingyu/data/caiyidong/dataset/MGIT/attribute/groundtruth/001.txt
        # data2/caiyidong/dataset/MGIT/attribute/groundtruth/001.txt

        ground_truth_rect = load_text(str(anno_path), delimiter=",", dtype=np.float64)

        nlp_path = "/home/luoxingyu/data/caiyidong/DFTrack/lib/test/evaluation/MGIT_nlp/{}.xlsx".format(sequence_name)
        nlp_tab = pd.read_excel(nlp_path)
        nlp_rect = nlp_tab.iloc[:, [14]].values
        nlp_rect = nlp_rect[-1, 0]

        # nlp_rect = nlp_rect[-3, 0]

        # nlp_rect = nlp_rect[0, 0]

        frames_path = r"{}/{}/{}/{}/{}_{}".format(
            self.base_path, "data", self.split, sequence_name, "frame", sequence_name
        )
        frame_list = [
            frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")
        ]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]


        return Sequence(
            sequence_name,
            frames_list,
            "videocube",
            ground_truth_rect.reshape(-1, 4),
            object_class=None,
            target_visible=None,
            language_query=nlp_rect,
        )

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        path = r"{}/{}/{}_list.txt".format(self.base_path, "data", "test")
        with open(path) as f:
            sequence_list = f.read().splitlines()

        return sequence_list
