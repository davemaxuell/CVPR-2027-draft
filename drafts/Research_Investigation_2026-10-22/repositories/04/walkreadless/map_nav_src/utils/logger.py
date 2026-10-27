import os
import sys
import math
import time
import csv
import torch
from thop import profile
from collections import OrderedDict


def flops_count(args, vln_bert, bs, txt_lens, global_lens, local_lens, candidate_lens, obj_size = 0, h_dim=768, irrelevant_words_id = None):
    
    # sanity check that candidate_lens is not larger than local_lens
    if candidate_lens + 2 > local_lens: # with <STOP> <MEM> tokens taken into account
        candidate_lens = local_lens - 2

    # set this to be true to avoid problems from random input.
    args.efficiency_count = True

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

    lan_input = {
        'txt_ids': txt_ids,
        'txt_masks': txt_masks,
        'instr_z_direction_features': instr_z_direction_features,
        'instr_z_direction_pzs': instr_z_direction_pzs,
        'instr_z_landmark_features': instr_z_landmark_features,
        'instr_z_landmark_pzs': instr_z_landmark_pzs,
        'front_txt_embeds': front_txt_embeds,
        'seq_lengths': seq_lengths,
        'irrelevant_words_id': irrelevant_words_id,
    }

    lan_macs, lan_params = profile(vln_bert, (mode, lan_input))
    lan_gflops = lan_macs/(10**9)

    # '''panorama'''
    local_pano_lens = 36
    view_img_fts = torch.rand(bs,local_pano_lens,h_dim).cuda()
    obj_img_fts = None
    loc_fts = torch.rand(bs,local_pano_lens + obj_size,7).cuda()
    nav_types = torch.randint(0,2,size=(bs,local_pano_lens + obj_size)).cuda()
    view_lens = torch.ones(bs,dtype=torch.int).cuda()*local_pano_lens

    z_img_features = torch.rand(bs,50,h_dim).cuda()
    z_img_pzs = torch.rand(bs,50,1).cuda()

    cand_vpids =  torch.randint(0,2,size=(bs, candidate_lens)).cuda()
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
    
    if args.dataset == 'reverie':
    
        obj_img_fts = torch.rand(bs,obj_size,h_dim).cuda()
        obj_lens = torch.tensor([obj_size for i in range(bs)]).cuda()
        obj_names = torch.randint(0, 2, size=(bs, obj_size), dtype=torch.int64).cuda()
        pan_input.update({
        'reverie_obj_img_fts': obj_img_fts,
        'reverie_obj_lens':    obj_lens,
        'reverie_obj_names':   obj_names,
        })

    pan_macs, pan_params = profile(vln_bert, (mode, pan_input))
    pan_gflops = pan_macs / (10 ** 9)

    mode = 'navigation'

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
    vp_cand_vpids = torch.randint(0, 2, size=(bs, candidate_lens+2), dtype=torch.bool).cuda() # copy from gmap_visited_masks
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

    return lan_gflops, pan_gflops, nav_gflops

    # print("lan_gflops",lan_gflops)
    # raise Exception


def write_results_to_file(results, file_path):
    file_exists = os.path.isfile(file_path)
    # Open the file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(results.keys())
        
        # Write the data
        writer.writerow(results.values())

def write_results(env_name, args, score_summary, agent):
    result_dict = {}
    result_dict['env_name'] = env_name
    result_dict['token_pruned_each_layer_pano'] = args.crop_vp_len_pano
    result_dict['action_steps'] = score_summary['action_steps']
    result_dict['sr'] = score_summary['sr']
    result_dict['spl'] = score_summary['spl']
    result_dict['vp_embeds_tokens'] = sum(agent.token_stats['pano_img_size'])/len(agent.token_stats['pano_img_size'])
    result_dict['txt_embeds_tokens'] = sum(agent.token_stats['text_size'])/len(agent.token_stats['text_size'])
    result_dict['gmap_size'] = sum(agent.token_stats['gmap_size'])/len(agent.token_stats['gmap_size'])
    result_dict['gmap_pruning_t_threshold'] = args.gmap_pruning_t_threshold
    result_dict['lan_gflops'] = score_summary['lan_gflops']
    result_dict['pan_gflops'] = score_summary['pan_gflops']
    result_dict['nav_gflops'] = score_summary['nav_gflops']
    result_dict['vln_bert_size'] = score_summary['vln_bert_size']
    result_dict['total_flops'] = score_summary['lan_gflops'] + (score_summary['pan_gflops'] + score_summary['nav_gflops']) * score_summary['action_steps']

    return result_dict

def write_results_reverie(env_name, args, score_summary, agent):
    result_dict = {}
    result_dict['env_name'] = env_name
    result_dict['token_pruned_each_layer_pano'] = args.crop_vp_len_pano
    result_dict['action_steps'] = score_summary['action_steps']
    result_dict['sr'] = score_summary['sr']
    result_dict['spl'] = score_summary['spl']
    result_dict['rgs'] = score_summary['rgs']
    result_dict['rgspl'] = score_summary['rgspl']
    result_dict['vp_embeds_tokens'] = sum(agent.token_stats['pano_img_size'])/len(agent.token_stats['pano_img_size'])
    result_dict['txt_embeds_tokens'] = sum(agent.token_stats['text_size'])/len(agent.token_stats['text_size'])
    result_dict['gmap_size'] = sum(agent.token_stats['gmap_size'])/len(agent.token_stats['gmap_size'])
    result_dict['gmap_pruning_t_threshold'] = args.gmap_pruning_t_threshold
    result_dict['lan_gflops'] = score_summary['lan_gflops']
    result_dict['pan_gflops'] = score_summary['pan_gflops']
    result_dict['nav_gflops'] = score_summary['nav_gflops']
    result_dict['vln_bert_size'] = score_summary['vln_bert_size']
    result_dict['total_flops'] = score_summary['lan_gflops'] + (score_summary['pan_gflops'] + score_summary['nav_gflops']) * score_summary['action_steps']

    return result_dict

# def get_results_dict_reverie(env_name, args, score_summary, agent):
#     result_dict = {}
#     result_dict['env_name'] = env_name
#     result_dict['token_pruned_each_layer_pano'] = args.crop_vp_len_pano
#     result_dict['action_steps'] = score_summary['action_steps']
#     result_dict['sr'] = score_summary['sr']
#     result_dict['spl'] = score_summary['spl']
#     result_dict['rgs'] = score_summary['rgs']
#     result_dict['rgspl'] = score_summary['rgspl']
#     result_dict['vp_embeds_tokens'] = sum(agent.token_stats['pano_img_size'])/len(agent.token_stats['pano_img_size'])
#     result_dict['txt_embeds_tokens'] = sum(agent.token_stats['text_size'])/len(agent.token_stats['text_size'])
#     result_dict['gmap_size'] = sum(agent.token_stats['gmap_size'])/len(agent.token_stats['gmap_size'])
#     result_dict['gmap_pruning_t_threshold'] = args.gmap_pruning_t_threshold
    

#     return result_dict

def write_to_record_file(data, file_path, verbose=True):
    if verbose:
        print(data)
        print("file_path",file_path)
    
    record_file = open(file_path, 'a')
    record_file.write(data+'\n')
    record_file.close()

    
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

class Timer:
    def __init__(self):
        self.cul = OrderedDict()
        self.start = {}
        self.iter = 0

    def reset(self):
        self.cul = OrderedDict()
        self.start = {}
        self.iter = 0

    def tic(self, key):
        self.start[key] = time.time()

    def toc(self, key):
        delta = time.time() - self.start[key]
        if key not in self.cul:
            self.cul[key] = delta
        else:
            self.cul[key] += delta

    def step(self):
        self.iter += 1

    def show(self):
        total = sum(self.cul.values())
        for key in self.cul:
            print("%s, total time %0.2f, avg time %0.2f, part of %0.2f" %
                  (key, self.cul[key], self.cul[key]*1./self.iter, self.cul[key]*1./total))
        print(total / self.iter)


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
