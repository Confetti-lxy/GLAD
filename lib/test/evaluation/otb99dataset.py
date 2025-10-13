import os

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class OTB99Dataset(BaseDataset):
    """ OTB-2015 dataset
    Publication:
        Object Tracking Benchmark
        Wu, Yi, Jongwoo Lim, and Ming-hsuan Yan
        TPAMI, 2015
        http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf
    Download the dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.otb99_path
        self.sequence_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence):
        sequence_path = '{}/{}'.format("OTB_videos", sequence)
        nz = 4
        ext = "jpg"
        start_frame = 1
        # special sequence to deal with
        if sequence == "BlurCar1":
            start_frame = 247
        elif sequence == "BlurCar3":
            start_frame = 3
        elif sequence == "BlurCar4":
            start_frame = 18
        elif sequence == "Board":
            nz = 5
        end_frame = len(os.listdir(os.path.join(self.base_path, "OTB_videos", sequence, 'img')))

        init_omit = 0

        frames = ['{base_path}/{sequence_path}/img/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                               sequence_path=sequence_path,
                                                                               frame=frame_num,
                                                                               nz=nz, ext=ext) for frame_num in
                  range(start_frame + init_omit, end_frame + 1)]
        anno_path = '{}/{}/{}'.format(self.base_path, sequence_path, "groundtruth_rect.txt")

        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        nlp_path = '{}/{}/{}'.format(self.base_path, "OTB_query_test", "{}.txt".format(sequence))
        nlp_rect = load_text(str(nlp_path), delimiter=',', dtype=str)
        nlp_rect = str(nlp_rect)
        return Sequence(sequence, frames, 'otb', ground_truth_rect[init_omit:, :],
                        object_class=sequence, language_query=nlp_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_info_list(self):
        sequence_list = ["Biker",
                         "Bird1",
                         "Bird2",
                         "BlurBody",
                         "BlurCar1",
                         "BlurCar2",
                         "BlurCar3",
                         "BlurCar4",
                         "BlurFace",
                         "BlurOwl",
                         "Board",
                         "Bolt2",
                         "Box",
                         "Car1",
                         "Car2",
                         "Car24",
                         "Coupon",
                         "Crowds",
                         "Dancer",
                         "Dancer2",
                         "Diving",
                         "Dog",
                         "DragonBaby",
                         "Girl2",
                         "Gym",
                         "Human2",
                         "Human3",
                         "Human4",
                         "Human5",
                         "Human6",
                         "Human7",
                         "Human8",
                         "Human9",
                         "Jump",
                         "KiteSurf",
                         "Man",
                         "Panda",
                         "RedTeam",
                         "Rubik",
                         "Skater",
                         "Skater2",
                         "Skating2-1",
                         "Skating2-2",
                         "Surfer",
                         "Toy",
                         "Trans",
                         "Twinnings",
                         "Vase", ]
        return sequence_list