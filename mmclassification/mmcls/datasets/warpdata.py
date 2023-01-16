import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class WARPDATA(BaseDataset):
    CLASSES = ["single_white_solid","single_white_dotted","single_yellow_solid","single_yellow_dotted","double_white_solid",\
            "double_yellow_solid","double_yellow_dotted","double_white_solid_dotted","double_white_dotted_solid"]
    def load_annotations(self):
        assert isinstance(self.ann_file, str)
        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos