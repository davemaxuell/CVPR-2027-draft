

from lavis.models import load_model_and_preprocess
import torch
from PIL import Image
import os
import json
from tqdm import tqdm
from GPT.api import gpt_infer_description


class BLIP_PanoManager(object):
    def __init__(self, args):

        self.args = args
        # setup device to use
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=self.device)
        self.image_root = "/egr/research-hlr2/zhan1624/mapgpt/observations"
        self.write_path = "/localscratch/zhan1624/MapGPT/resources/blip_pano_description.json"

    def get_pano_descriptions(self, pano_desp, scan, viewpoint):
        image_dir = self.image_root+"/"+scan+"/"+viewpoint
        img_dict = {}
        all_images = []
        keys = []
        for each_image in tqdm(os.listdir(image_dir)):

        # prepare the image
            raw_image = Image.open(image_dir+"/"+each_image).convert("RGB")
            image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
            res = self.model.generate({"image": image, "prompt": "Describe this indoor scene in details with two sentence. Please Describe the most obvious objects."})
            img_dict[each_image.split('.')[0]] = res[0]
        if scan not in pano_desp:
            pano_desp[scan] = {}
        pano_desp[scan][viewpoint] = {}
        pano_desp[scan][viewpoint] = img_dict
        
        with open(self.write_path, 'w') as file:
            json.dump(pano_desp, file, indent=4)



class GPT_PanoManager(object):
    def __init__(self, args):

        self.args = args
        # setup device to use
        self.image_root = "/egr/research-hlr2/zhan1624/mapgpt/observations"
        self.desp_prompt = """Describe indoor scene in each iamge in details with two sentence. Please Describe the most obvious objects. Structure the response in JSON format, using keys like 'image1', 'image2', and their respective detailed descriptions."""
        self.write_path = "/localscratch/zhan1624/MapGPT/resources/gpt_pano_description.json"

    def get_pano_descriptions(self, pano_desp, scan, viewpoint):
        image_dir = self.image_root+"/"+scan+"/"+viewpoint
        img_dict = {}
        res = {}
        while len(res) != 36:
            res = gpt_infer_description("", self.desp_prompt,[image_dir+"/"+each_file for each_file in os.listdir(image_dir)],
                                                        self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})[0]
            res = json.loads(res)
            
        for each_key, each_value in res.items():
            img_dict[each_key[5:]] = each_value
        if scan not in pano_desp:
            pano_desp[scan] = {}
        pano_desp[scan][viewpoint] = {}
        pano_desp[scan][viewpoint] = img_dict
        
        with open(self.write_path, 'w') as file:
            json.dump(pano_desp, file, indent=4)

        


