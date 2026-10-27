import json
from tqdm import tqdm
from .base_dataset import BaseDataset

class RXRDataset(BaseDataset):
    name = "rxr"

    def load_data(self, anno_file):
        with open(anno_file, "r") as f:
            data = json.load(f)

        new_data = []

        data = tqdm(data, desc="Loading data")
        for i, item in enumerate(data):
            new_item = dict(item)
            new_item['raw_idx'] = i
            new_item['instr_id'] = f"{self.name}_{item['path_id']}"

            new_item['instruction'] = item["instruction"]
            new_item['instr_encoding'] = item['instr_encoding']


            new_item['data_type'] = 'rxr'

            new_data.append(new_item)

        gt_trajs = {
            x['instr_id']: (x['scan'], x['path']) \
            for x in new_data if len(x['path']) > 1
        }    

        return new_data, gt_trajs