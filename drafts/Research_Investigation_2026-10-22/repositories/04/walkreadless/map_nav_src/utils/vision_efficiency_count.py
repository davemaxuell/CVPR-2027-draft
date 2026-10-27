
# This is to count the size of the parameters of the model
import os,sys
root_path = os.getcwd()
current_path = os.path.join(root_path,'map_nav_src_analysis_5')
sys.path.append(root_path)
sys.path.append(current_path)

import torch
import torch.nn as nn
import numpy as np
from r2r.parser import parse_args
from models.model import VLNBert
from models import *
from thop import profile

def calculate_GFLOPS(model):
    txt_embeds = torch.rand(bs,txt_lens,h_dim).cuda()
    txt_masks = torch.rand(bs,txt_lens).cuda()

    vp_img_embeds = torch.rand(bs,local_lens,h_dim).cuda()
    vp_masks = torch.randint(1, 2, size=(bs, local_lens), dtype=torch.bool).cuda()
    
    vp_nav_masks = torch.randint(0, 2, size=(bs, local_lens), dtype=torch.bool).cuda()
    vp_obj_masks = None
    vp_cand_vpids = torch.randint(0, 2, size=(bs, global_lens), dtype=torch.bool).cuda() # copy from gmap_visited_masks

    vp_img_ids = []
    for i in range(bs):
        vp_img_ids_i = []
        for j in range(36):
            vp_img_ids_i.append(j)
        vp_img_ids.append(vp_img_ids_i)
    vp_img_ids = torch.tensor(vp_img_ids).cuda()

    # self, q_embeds, q_masks, kv_embeds, kv_masks, graph_sprels=None, output_attentions = None, saliency_method = None, r = None, method = None, \
                # cand_vpids = None, vp_img_ids = None, attentive_txt_indexes = None, if_skip_attn_pruning = False, if_token_attn_pruning = False

    nav_macs, nav_params = profile(local_encoder, (vp_img_embeds, vp_masks, txt_embeds, txt_masks, None, None, None, args.crop_vp_len, args.vp_crop_method, \
                                                   vp_cand_vpids, vp_img_ids, None, True, True))
    nav_gflops = nav_macs * 2 / (10 ** 9)

    return nav_gflops
    


if __name__ == '__main__':
    '''
    bs: batch size
    txt_lens: the length of the instruction
    local_lens: the number of the local views
    global_lens: the number of the global views
    h_dim: the dimension of the hidden state
    k_neighbors: the number of the neighbors for each view in the panoramic image
    n_steps: the number of the steps for the navigation
    '''
    args = parse_args()

    bs = args.batch_size
    txt_lens = 50
    h_dim = 768

    local_lens = 38
    global_lens = 6

    '''Parameter count'''
    vln_bert = VLNBert(args).cuda()
    vln_bert_size = sum([x.numel() for x in vln_bert.parameters()])
    local_encoder = vln_bert.vln_bert.local_encoder.encoder
    
    crop_vp_lens =  list(range(0,32,1))
    result = {}
    # print("crop_vp_lens",crop_vp_lens)
    for lens in crop_vp_lens:
        args.crop_vp_len = lens
        print("args.crop_vp_len: ",lens)
        nav_gflops = calculate_GFLOPS(local_encoder)
        result[lens] = round(nav_gflops,2)
        print('Navigation Module GFLOPs: %.3f' % (nav_gflops))
        print("")
    print("result",result)
        # print('*****')
        # print('Total Parameters: %.3f M.'%(vln_bert_size/1e6))
    # vp_embeds = local_encoder(vp_img_embeds, vp_masks, txt_embeds, txt_masks, r = args.crop_vp_len, method = args.vp_crop_method, \
    #                                            cand_vpids = vp_cand_vpids, vp_img_ids = vp_img_ids, \
    #                                             if_skip_attn_pruning = False, if_token_attn_pruning = False)
    

    
