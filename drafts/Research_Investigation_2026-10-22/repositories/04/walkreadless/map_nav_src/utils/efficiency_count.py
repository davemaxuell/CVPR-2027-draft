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
import csv
from tqdm import tqdm

# local_lens_dict_r2r = {
#         0:38, 8:22, 10:18, 13:16
#     }

# r2r
# # tome
# local_lens_dict_r2r = {
#         0:38, 1:36, 2:34, 3:32,
#         4:30, 5:28, 6:26, 7:24,
#         8:22, 9:20, 10:18, 11:17,
#         12:16, 13:15, 14:15, 15:14,
#         16: 14, 17: 14, 18:14, 19:14, 20:14
#     }

# others
local_lens_dict_r2r = {
        0:38, 1:36, 2:34, 3:32,
        4:30, 5:28, 6:26, 7:24,
        8:22, 9:20, 10:18, 11:16,
        12:14, 13:12, 14:10, 15:8,
        16: 7, 17: 6, 18:6, 19:6, 20:6
    }

gmap_lens_dict_r2r = {
        0:14, 15:14, 14: 14, 13:14, 12:14,
        11:13, 10: 13, 9: 13, 8: 13,
        7: 12, 6: 12, 5: 11, 4: 10, 3:10, 2: 9
    }

# local_lens_dict_r2r = {
#         0:38, 5: 28, 8:22, 13:12,17:4
#     }

# gmap_lens_dict_r2r = {
#         0:14, 15:14, 14: 14, 13:14, 12:14,
#         11:13, 10: 13, 9: 13, 8: 13,
#         7: 12, 6: 12, 5: 11, 4: 10, 3:10
#     }

# # rxr
local_lens_dict_rxr = {
        0:38, 1:36, 2:34, 3:32,
        4:30, 5:28, 6:26, 7:24,
        8:22, 9:20, 10:18, 11:16,
        12:14, 13:12, 14:10, 15:9,
        16:7, 17:6, 18:6, 19:6, 20:6
    }

# tome
# local_lens_dict_rxr = {
#         0:38, 5: 28, 8:22, 13:15, 20:13
#     }
gmap_lens_dict_rxr = {
        0: 19, 15:18, 14:17, 13:17, 12:17,
        11:16, 10: 16, 9: 16, 8: 15,
        7:15, 6:14, 5:13, 4:13, 3:12
    }

# reverie

# # tome
# local_lens_dict_reverie = {
#         0:42, 1:40, 2:38, 3:36,
#         4:34, 5:32, 6:30, 7:28,
#         8:26, 9:24, 10:22, 11:20,
#         12:19, 13:19, 14:18, 15:18,
#         16:17, 17:17, 18:17, 19:17, 20:17
#     }

# others
# local_lens_dict_reverie = {
#         0:42, 1:40, 2:38, 3:36,
#         4:34, 5:32, 6:30, 7:28,
#         8:26, 9:24, 10:22, 11:20,
#         12:18, 13:16, 14:14, 15:12,
#         16:10, 17:10, 18:10, 19:10, 20:10
#     }

# gmap_lens_dict_reverie = {
#         0: 16, 15: 16, 14: 16,
#         13: 16, 12:15, 11:15, 
#         10: 15, 9: 15, 8: 14,
#         7:14, 6:13, 5:12, 4:12, 3:11
#     }


def GFLOPs_count(bs, txt_lens, global_lens, local_lens, txt_pruning_rate, visual_token_pruning_n, gmap_token_pruning_n, h_dim=768,dataset=None):


    # '''Language'''
    txt_ids = torch.randint(0,2000,size=(bs,txt_lens)).cuda()
    txt_masks = torch.randint(0, 2, size=(bs, txt_lens), dtype=torch.bool).cuda()
    instr_z_direction_features = torch.rand(bs,35,h_dim).cuda()
    instr_z_direction_pzs = torch.rand(bs,35,1).cuda()
    instr_z_landmark_features = torch.rand(bs,39,h_dim).cuda()
    instr_z_landmark_pzs = torch.rand(bs,39,1).cuda()
    seq_lengths = [len(txt_ids[i]) for i in range(bs)]
    front_txt_embeds = torch.rand(bs,24,h_dim)
    mode = 'language'

    txt_gflops_dict = {}

    args.txt_pruning_rate = txt_pruning_rate
    lan_input = {
        'txt_ids': txt_ids,
        'txt_masks': txt_masks,
        'instr_z_direction_features': instr_z_direction_features,
        'instr_z_direction_pzs': instr_z_direction_pzs,
        'instr_z_landmark_features': instr_z_landmark_features,
        'instr_z_landmark_pzs': instr_z_landmark_pzs,
        'front_txt_embeds': front_txt_embeds,
        'seq_lengths': seq_lengths,
    }

    lan_macs, lan_params = profile(vln_bert, (mode, lan_input))
    lan_gflops = lan_macs/(10**9)

    # print("lan_gflops",lan_gflops)
    # raise Exception
    

    # '''panorama'''
    local_pano_lens = 36
    view_img_fts = torch.rand(bs,local_pano_lens,h_dim).cuda()
    obj_img_fts = None
    loc_fts = torch.rand(bs,local_pano_lens,7).cuda()
    nav_types = torch.randint(0,2,size=(bs,local_pano_lens)).cuda()
    view_lens = torch.ones(bs,dtype=torch.int).cuda()*local_pano_lens

    z_img_features = torch.rand(bs,50,h_dim).cuda()
    z_img_pzs = torch.rand(bs,50,1).cuda()

    cand_vpids =  torch.randint(0,2,size=(bs, candidate_each_step)).cuda()
    vp_img_ids = torch.tensor([list(range(0, 36)) for i in range(bs)])

    mode = 'panorama'

    pan_input = {
        'view_img_fts': view_img_fts,
        'obj_img_fts': obj_img_fts,
        'loc_fts': loc_fts,
        'nav_types': nav_types,
        'cand_vpids': cand_vpids,
        'view_lens': view_lens,
        'z_img_features': z_img_features,
        'z_img_pzs': z_img_pzs,
        'vp_img_ids': vp_img_ids,
        }


    args.crop_vp_len_pano = visual_token_pruning_n
    pan_macs, pan_params = profile(vln_bert, (mode, pan_input))
    pan_gflops = pan_macs / (10 ** 9)

    # print(f'{args.crop_vp_len_pano} tokens pruned each layer: Panorama Module GFLOPs: %.3f' % (pan_gflops))
    # print('*****')
    # raise Exception
    

    '''navigation'''

    
    
    nav_gflops_dict = {}


    mode = 'navigation'
    # local_lens = 39

    if dataset == 'r2r':
        local_lens = local_lens_dict_r2r[visual_token_pruning_n]
        global_lens = gmap_lens_dict_r2r[gmap_token_pruning_n]
        txt_lens = round(txt_lens * (1 - txt_pruning_rate))
    elif dataset == 'rxr':
        local_lens = local_lens_dict_rxr[visual_token_pruning_n]
        global_lens = gmap_lens_dict_rxr[gmap_token_pruning_n]
        # local_lens = local_lens_dict_reverie[visual_token_pruning_n]
        # global_lens = gmap_lens_dict_reverie[gmap_token_pruning_n]
        txt_lens = round(txt_lens * (1 - txt_pruning_rate))
    else:
        raise NotImplementedError

    # print("local_lens",local_lens)
    # print("global_lens",global_lens)
    args.crop_vp_len_nav = 0
    

        
    txt_embeds = torch.rand(bs,txt_lens,h_dim).cuda()
    txt_masks = torch.rand(bs,txt_lens).cuda()
    gmap_img_embeds = torch.rand(bs,global_lens,h_dim).cuda()
    gmap_step_ids = torch.randint(0, 2, size=(bs, global_lens), dtype=torch.int).cuda()
    gmap_pos_fts = torch.rand(bs,global_lens,7).cuda()
    gmap_masks = torch.randint(0, 2, size=(bs, global_lens), dtype=torch.bool).cuda()
    gmap_pair_dists = torch.rand(bs,global_lens,global_lens).cuda()
    gmap_visited_masks = torch.randint(0, 2, size=(bs, global_lens), dtype=torch.bool).cuda()
    gmap_vpids = torch.randint(0, 2, size=(bs, global_lens), dtype=torch.bool).cuda() # copy from gmap_visited_masks

    front_vp_feats = torch.rand(bs,24,h_dim).cuda()
    front_gmap_feats = torch.rand(bs,24,h_dim).cuda()

    vp_img_embeds = torch.rand(bs,local_lens,h_dim).cuda()
    vp_pos_fts = torch.rand(bs,local_lens,14).cuda()
    vp_masks = torch.randint(0, 2, size=(bs, local_lens), dtype=torch.bool).cuda()
    vp_nav_masks = torch.randint(0, 2, size=(bs, local_lens), dtype=torch.bool).cuda()
    vp_obj_masks = None
    vp_cand_vpids = torch.randint(0, 2, size=(bs, candidate_each_step), dtype=torch.bool).cuda() # copy from gmap_visited_masks
    vp_img_ids = torch.tensor([list(range(0, local_lens - 2)) for i in range(bs)]) 

    nav_input = {
    'txt_embeds': txt_embeds,
    'txt_masks': txt_masks,
    'gmap_img_embeds': gmap_img_embeds,
    'gmap_step_ids': gmap_step_ids,
    'gmap_pos_fts': gmap_pos_fts,
    'gmap_masks': gmap_masks,
    'gmap_pair_dists': gmap_pair_dists,
    'gmap_visited_masks': gmap_visited_masks,
    'gmap_vpids': gmap_vpids,
    'vp_img_embeds': vp_img_embeds,
    'vp_pos_fts': vp_pos_fts,
    'vp_masks': vp_masks,
    'vp_nav_masks': vp_nav_masks,
    'vp_obj_masks': vp_obj_masks,
    'vp_cand_vpids': vp_cand_vpids,
    'front_vp_feats': front_vp_feats,
    'front_gmap_feats': front_gmap_feats,
    "vp_img_ids": vp_img_ids, 
    }

    nav_macs, nav_params = profile(vln_bert, (mode, nav_input))
    nav_gflops = nav_macs / (10 ** 9)



    # print('Navigation Module GFLOPs %.3f' % (nav_gflops))
    # print('*****')
    # print("nav_gflops",nav_gflops)
    # raise Exception

    return {"lan_gflops":lan_gflops,"pan_gflops":pan_gflops,"nav_gflops":nav_gflops}
    # total_GFLOPs = nav_gflops

    max_memory = torch.cuda.max_memory_allocated(device = 'cuda')
    print("max_memory",max_memory)
    return total_GFLOPs
    
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

    
    '''FLOPs count'''
    # Default definition
    
    args = parse_args()

    # # Reverie setting
    # args.dataset = 'rxr'
    # # args.vp_crop_method = 'attn_scores'
    # # args.txt_prune_strat = 'attn_scores'
    # args.vp_crop_method = 'tome'
    # # raise Exception("vp_crop_method")
    # args.txt_prune_strat = 'attn_scores'
    # bs = 1
    # txt_lens = 21
    # local_lens = 42
    # global_lens = 16
    # candidate_each_step = 6
    # h_dim = 768
    # k_neighbors = 18
    # n_steps = 7

    # print("args.vp_crop_method",args.vp_crop_method)
    # print("args.txt_prune_strat",args.txt_prune_strat)

    # # RxR setting
    # args.dataset = 'rxr'
    # args.vp_crop_method = 'attn_scores'
    # args.txt_prune_strat = 'attn_scores'
    # bs = 1
    # txt_lens = 128
    # local_lens = 38
    # global_lens = 19
    # candidate_each_step = 6
    # h_dim = 768
    # k_neighbors = 18
    # n_steps = 7

    ############################################################################################################
    
    # R2R setting
    args.vp_crop_method = 'attn_scores'
    args.txt_prune_strat = 'attn_scores'
    args.dataset = 'r2r'
    bs = 1
    txt_lens = 32
    local_lens = 38
    global_lens = 14
    candidate_each_step = 6
    h_dim = 768
    k_neighbors = 18
    n_steps = 7

    result_out = []

    # r2r our method  
    settings = []           
    # for gtpn in range(15,2,-1):   
    # settings = \
    #     [{
    #         "txt_pruning_rate":0.0,
    #         "visual_token_pruning_n":0,
    #         "gmap_token_pruning_n":0,
    #     },
    #     {   # BGP
    #         "txt_pruning_rate":0, # VPP
    #         "visual_token_pruning_n":12, # BGP
    #         "gmap_token_pruning_n":0, # BTP
    #     },
    #     {   # BTP
    #         "txt_pruning_rate":0.0, # VPP
    #         "visual_token_pruning_n":0, # BGP
    #         "gmap_token_pruning_n":4, # BTP
    #     },
    #     {   # VPP
    #         "txt_pruning_rate":0.5, # VPP
    #         "visual_token_pruning_n":0, # BGP
    #         "gmap_token_pruning_n":0, # BTP
    #     },
    #     {   # BGP + BTP
    #         "txt_pruning_rate":0.0, # VPP
    #         "visual_token_pruning_n":12, # BTP
    #         "gmap_token_pruning_n":4, # BTP
    #     },
    #     {   # BGP + VPP
    #         "txt_pruning_rate":0.5, # VPP
    #         "visual_token_pruning_n":12, # BGP
    #         "gmap_token_pruning_n":0, # BTP
    #     },
    #     {   # BTP + VPP
    #         "txt_pruning_rate":0.5, # VPP
    #         "visual_token_pruning_n":0, # BGP
    #         "gmap_token_pruning_n":4, # BTP
    #     },
    #     {   # BGP + BTP + VPP
    #         "txt_pruning_rate":0.5, # VPP
    #         "visual_token_pruning_n":12, # BGP
    #         "gmap_token_pruning_n":4, # BTP
    #     },
    #     ]

    gtpns = [0]
    vtpns = [0]
    tprs = [0.0]
    for gpt in gtpns:
        for vtpn in vtpns:
            for tpr in tprs:    
                settings += \
                    [{
                        "txt_pruning_rate":tpr,
                        "visual_token_pruning_n":vtpn,
                        "gmap_token_pruning_n":gpt,
                    },]
    # settings = \
    #     [{
    #         "txt_pruning_rate":0.0,
    #         "visual_token_pruning_n":0,
    #         "gmap_token_pruning_n":0,
    #     },
    #     {   # BGP + BTP + VPP
    #         "txt_pruning_rate":0.3, # VPP
    #         "visual_token_pruning_n":8, # BGP
    #         "gmap_token_pruning_n":0, # BTP
    #     },
    #     {   # BGP + BTP + VPP
    #         "txt_pruning_rate":0.4, # VPP
    #         "visual_token_pruning_n":10, # BGP
    #         "gmap_token_pruning_n":0, # BTP
    #     },
    #     {   # BGP + BTP + VPP
    #         "txt_pruning_rate":0.5, # VPP
    #         "visual_token_pruning_n":13, # BGP
    #         "gmap_token_pruning_n":0, # BTP
    #     },
    #     ]

    # for vtpn in [0,5,8,13,17]:
    # for gtpn in [15,9,7,5,3]:
    #     settings += \
    #         [{
    #             "txt_pruning_rate":0.0,
    #             "visual_token_pruning_n":0,
    #             "gmap_token_pruning_n":gtpn,
    #         },]
    

    '''Parameter count'''
    vln_bert = VLNBert(args).cuda()
    vln_bert_size = sum([x.numel() for x in vln_bert.parameters()])
    print('Total Parameters: %.3f M.'%(vln_bert_size/1e6))

    for setting in settings:
        tpr = setting["txt_pruning_rate"]
        vtpn = setting["visual_token_pruning_n"]
        gtpn = setting["gmap_token_pruning_n"]

        gflops = GFLOPs_count(bs,txt_lens,global_lens,local_lens,tpr,vtpn,gtpn,h_dim,dataset = args.dataset)
        gflops_dict = {
            "txt_pruning_rate":tpr,
            "visual_token_pruning_n":vtpn,
            "gmap_token_pruning_n":gtpn,
            "lan_gflops": gflops['lan_gflops'],
            "pan_gflops": gflops['pan_gflops'],
            "nav_gflops": gflops['nav_gflops'],
        }

        result_out.append(gflops_dict)

    csv_file = 'r2r_gflops_results.csv'
    csv_columns = ['txt_pruning_rate', 'visual_token_pruning_n', 'gmap_token_pruning_n', 'lan_gflops', 'pan_gflops', 'nav_gflops']

    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in result_out:
                writer.writerow(data)
    except IOError:
        print("I/O error")
    raise Exception("!!!!!!!")
    ############################################################################################################
    # # RxR setting
    # args.dataset = 'rxr'
    # args.vp_crop_method = 'attn_scores'
    # args.txt_prune_strat = 'attn_scores'
    # bs = 1
    # txt_lens = 128
    # local_lens = 38
    # global_lens = 19
    # candidate_each_step = 6
    # h_dim = 768
    # k_neighbors = 18
    # n_steps = 7

    # result_out = []

    # # rxr our method  
    # settings = []           
    # # for gtpn in range(15,2,-1):   
    # #     settings += \
    # #         [{
    # #             "txt_pruning_rate":0.0,
    # #             "visual_token_pruning_n":12,
    # #             "gmap_token_pruning_n":gtpn,
    # #         },]
    
    # for vtpn in range(21):
    #     settings += \
    #         [{
    #             "txt_pruning_rate":0.0,
    #             "visual_token_pruning_n":vtpn,
    #             "gmap_token_pruning_n":8,
    #         },]
        
    # for setting in settings:
    #     tpr = setting["txt_pruning_rate"]
    #     vtpn = setting["visual_token_pruning_n"]
    #     gtpn = setting["gmap_token_pruning_n"]

    #     gflops = GFLOPs_count(bs,txt_lens,global_lens,local_lens,tpr,vtpn,gtpn,h_dim,dataset = args.dataset)
    #     gflops_dict = {
    #         "txt_pruning_rate":tpr,
    #         "visual_token_pruning_n":vtpn,
    #         "gmap_token_pruning_n":gtpn,
    #         "lan_gflops": gflops['lan_gflops'],
    #         "pan_gflops": gflops['pan_gflops'],
    #         "nav_gflops": gflops['nav_gflops'],
    #     }

    #     result_out.append(gflops_dict)

    # csv_file = 'rxr_gflops_results.csv'
    # csv_columns = ['txt_pruning_rate', 'visual_token_pruning_n', 'gmap_token_pruning_n', 'lan_gflops', 'pan_gflops', 'nav_gflops']

    # try:
    #     with open(csv_file, 'w') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    #         writer.writeheader()
    #         for data in result_out:
    #             writer.writerow(data)
    # except IOError:
    #     print("I/O error")

    # ############################################################################################################

    # # Reverie setting
    # args.dataset = 'rxr'
    # args.vp_crop_method = 'attn_scores'
    # args.txt_prune_strat = 'attn_scores'
    # bs = 1
    # txt_lens = 21
    # local_lens = 42
    # global_lens = 16
    # candidate_each_step = 6
    # h_dim = 768
    # k_neighbors = 18
    # n_steps = 7

    # result_out = []

    # rxr our method  
    # settings = []           
    # for gtpn in range(15,2,-1):   
    #     settings += \
    #         [{
    #             "txt_pruning_rate":0.0,
    #             "visual_token_pruning_n":12,
    #             "gmap_token_pruning_n":gtpn,
    #         },]
    
    # for vtpn in range(21):
    #     settings += \
    #         [{
    #             "txt_pruning_rate":0.0,
    #             "visual_token_pruning_n":vtpn,
    #             "gmap_token_pruning_n":8,
    #         },]
        
    # for setting in settings:
    #     tpr = setting["txt_pruning_rate"]
    #     vtpn = setting["visual_token_pruning_n"]
    #     gtpn = setting["gmap_token_pruning_n"]

    #     gflops = GFLOPs_count(bs,txt_lens,global_lens,local_lens,tpr,vtpn,gtpn,h_dim,dataset = args.dataset)
    #     gflops_dict = {
    #         "txt_pruning_rate":tpr,
    #         "visual_token_pruning_n":vtpn,
    #         "gmap_token_pruning_n":gtpn,
    #         "lan_gflops": gflops['lan_gflops'],
    #         "pan_gflops": gflops['pan_gflops'],
    #         "nav_gflops": gflops['nav_gflops'],
    #     }

    #     result_out.append(gflops_dict)

    # csv_file = 'reverie_gflops_results.csv'
    # csv_columns = ['txt_pruning_rate', 'visual_token_pruning_n', 'gmap_token_pruning_n', 'lan_gflops', 'pan_gflops', 'nav_gflops']

    # try:
    #     with open(csv_file, 'w') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    #         writer.writeheader()
    #         for data in result_out:
    #             writer.writerow(data)
    # except IOError:
    #     print("I/O error")

    # for tpr in txt_pruning_rates:
    #     for vtpn in tqdm(visual_token_pruning_n_settings):
    #         for gtpn in gmap_token_pruning_n_settings:

    # # r2r attn scores             
    # settings = [
    #     {
    #         "txt_pruning_rate":0.0,
    #         "visual_token_pruning_n":0,
    #         "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.1,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.2,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.3,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.4,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.5,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.6,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.7,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.8,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.9,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    # ]

    

    # # reverie attn scores             
    # settings = [
    #     # {
    #     #     "txt_pruning_rate":0.0,
    #     #     "visual_token_pruning_n":0,
    #     #     "gmap_token_pruning_n":0,
    #     # },
    #     {
    #     "txt_pruning_rate":0.3,
    #     "visual_token_pruning_n":8,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.4,
    #     "visual_token_pruning_n":11,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.5,
    #     "visual_token_pruning_n":14,
    #     "gmap_token_pruning_n":0,
    #     },
    # ]

    # # reverie our method             
    # settings = [
    #     {
    #     "txt_pruning_rate":0.0,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.1,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.2,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.3,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.4,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.5,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.6,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.7,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.8,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.9,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    # ]

    # # rxr attn scores             
    # settings = [
    #     {
    #         "txt_pruning_rate":0.0,
    #         "visual_token_pruning_n":0,
    #         "gmap_token_pruning_n":0,
    #     },
    #     {
    #         "txt_pruning_rate":0.1,
    #         "visual_token_pruning_n":0,
    #         "gmap_token_pruning_n":0,
    #     },
    #     {
    #         "txt_pruning_rate":0.2,
    #         "visual_token_pruning_n":0,
    #         "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.3,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.4,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.5,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.6,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.7,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.8,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    #     {
    #     "txt_pruning_rate":0.9,
    #     "visual_token_pruning_n":0,
    #     "gmap_token_pruning_n":0,
    #     },
    # ]


    # # rxr our method
    # settings = [
    #     {
    #     "txt_pruning_rate":0.3,
    #     "visual_token_pruning_n":7,
    #     "gmap_token_pruning_n":8,
    #     },
    #     {
    #     "txt_pruning_rate":0.4,
    #     "visual_token_pruning_n":10,
    #     "gmap_token_pruning_n":8,
    #     },
    #     {
    #     "txt_pruning_rate":0.5,
    #     "visual_token_pruning_n":13,
    #     "gmap_token_pruning_n":8,
    #     },
    # ]

    
    

    # total_gflops = GFLOPs_count(bs,txt_lens,global_lens,local_lens,h_dim)

    print('Total Parameters: %.3f M.'%(vln_bert_size/1e6))