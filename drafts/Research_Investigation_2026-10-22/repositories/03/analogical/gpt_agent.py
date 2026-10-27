import sys
import numpy as np
from collections import defaultdict
from GPT.thinker_manager import OneStagePromptManager
#from GPT.pano_manager import GPT_PanoManager, BLIP_PanoManager
from agent_base import BaseAgent
from GPT.api import gpt_infer_action_both, gpt_infer_description, gpt_infer_action, gpt_infer_env
import json
from data_utils import add_data_to_json, process_img
from json.decoder import JSONDecodeError
from GPT.qwen_utils import qwen_infer, qwen_infer_desription
from GPT.gemini_utils import gemini_infer

# Configure logging

candi_des_path = "/localscratch/zhan1624/MapGPT/resources/compare_candi_description.json"
#candi_des_path = "/localscratch/zhan1624/MapGPT/resources/compare_candi_description_reverie.json"
pano_des_path = "/localscratch/zhan1624/MapGPT/resources/blip_pano_description.json"

with open(candi_des_path) as f_can, open(pano_des_path) as f_pano:
    candi_desp = json.load(f_can)
    pano_desp = json.load(f_pano)

class GPTNavAgent(BaseAgent):
    env_actions = {
        'left': (0, -1, 0),  # left
        'right': (0, 1, 0),  # right
        'up': (0, 0, 1),  # up
        'down': (0, 0, -1),  # down
        'forward': (1, 0, 0),  # forward
        '<end>': (0, 0, 0),  # <end>
        '<start>': (0, 0, 0),  # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0, model=None, processor=None):
        super().__init__(env)
        self.args = args
        self.rank = rank
        self.model = model
        self.processor = processor

        self._build_prompt_manager()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)
    
    def _build_prompt_manager(self):
        self.prompt_manager = OneStagePromptManager(self.args)
        #self.pano_manager = BLIP_PanoManager(self.args)
        print('Model version:', self.args.llm)

    def make_equiv_action(self, a_t, obs, traj=None):

        def take_action(i, name):
            if type(name) is int:       # Go to the next viewpoint
                self.env.env.sims[i].makeAction([name], [0], [0])
            else:                       # Adjust
                self.env.env.sims[i].makeAction(*self.env_actions[name])

        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = ob['candidate'][action]
                src_point = ob['viewIndex']
                trg_point = select_candidate['pointId']
                print(ob['viewpoint'], trg_point, select_candidate['viewpointId'])
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, 'down')
                    src_level -= 1
                while self.env.env.sims[i].getState()[0].viewIndex != trg_point:    # Turn right until the target
                    take_action(i, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[i].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, select_candidate['idx']) # j+1: idx for navigable location

                state = self.env.env.sims[i].getState()[0]
                if traj is not None:
                    traj[i]['path'].append([state.location.viewpointId])

    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        batch_size = len(obs)
        scan = obs[0]['scan']
        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
            'a_t': {},
            'map_log': [],
        } for ob in obs]
        map_log = []

        if traj[0]['instr_id'] in self.results:
            return [None]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        previous_angle = [{'heading': ob['heading'],
                               'elevation': ob['elevation']} for ob in obs]

        self.prompt_manager.history = ['' for _ in range(self.args.batch_size)]
        self.prompt_manager.nodes_list = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.node_imgs = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.graph = [{} for _ in range(self.args.batch_size)]
        self.prompt_manager.trajectory = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.planning = [["Navigation has just started, with no planning yet."] for _ in range(self.args.batch_size)]

        for t in range(self.args.max_action_len):
            if t == self.args.max_action_len:
                break
          

            cand_inputs = self.prompt_manager.make_action_prompt(obs, previous_angle)

            if self.args.response_format == 'str':
                nav_input = self.prompt_manager.make_r2r_prompts(cand_inputs=cand_inputs, obs=obs, t=t)
            elif self.args.response_format == 'json':
                nav_input = self.prompt_manager.make_r2r_json_prompts(cand_inputs=cand_inputs, obs=obs, t=t, previous_angle=previous_angle)
            else:
                raise NotImplemented

            image_list = self.prompt_manager.node_imgs[0]
            environment_prompts = nav_input["prompts"][0]
            print('-------------------- Environment Prompts --------------------')
            print(environment_prompts)

            if self.args.llm == 'gpt-4-vision-preview' and self.args.response_format == 'str':
                # GPT-4V only supports string mode output
                descript_output, tokens = gpt_infer(nav_input["task_description"], environment_prompts, image_list,
                                               self.args.llm, self.args.max_tokens)
                print('-------------------- Output --------------------')
                print(descript_output)
                nav_output = [nav_output]
                a_t = self.prompt_manager.parse_action(nav_output=nav_output,
                                                       only_options_batch=nav_input["only_options"],
                                                       t=t)
                self.prompt_manager.parse_planning(nav_output=nav_output)

            elif self.args.llm == 'gpt-4o' and self.args.response_format == 'json':
                if len(image_list) > 20:
                    # GPT-4o currently does not support queries with more than 20 images
                    a_t = [0]
                    print('Exceed image limit and stop!')
                else:
                    if scan not in candi_desp:
                        candi_desp[scan] = {}
                    if self.prompt_manager.trajectory[0][-1] not in candi_desp[scan]: 
                        node_list = cand_inputs['map_nodes'][0]
                        if image_list[0] == None:
                            node_list.insert(0, None)

                        new_image_list = []
                        for node in node_list:
                            if node == None:
                                continue
                            new_image_list.append(image_list[node])
                        remove_dup_img = list(dict.fromkeys(new_image_list))
                        tmp_image_desp = []
            
                        nav_output, tokens = gpt_infer_description(nav_input["img2txt_task"], nav_input["img2txt_prompts"][0], remove_dup_img,
                                                    self.args.llm, 2000, response_format={"type": "json_object"})

                        json_des_output = json.loads(nav_output)
                        tmp_image_desp = list(json_des_output.values())

                        if len(tmp_image_desp) != len(remove_dup_img):
                            print(obs[0]['instr_id'])
                        assert len(tmp_image_desp) == len(remove_dup_img)
                        
                        new_image_dict = dict(zip(remove_dup_img, tmp_image_desp))
                        if image_list[0] == None:
                            tmp_image_desp.insert(0, None)
                        for id, node in enumerate(node_list):
                            if node == None:
                                continue
                            image_names = image_list[node].split("/")
                            viewpoint, image_id = image_names[-2], int(image_names[-1].split(".")[0])
                            if viewpoint not in candi_desp[scan]:
                                candi_desp[scan][viewpoint] = {}
                            if str(image_id) not in candi_desp[scan][viewpoint]:
                                candi_desp[scan][viewpoint][str(image_id)] = new_image_dict[image_list[node]]
                                
                        add_data_to_json(candi_des_path, candi_desp)
                    
                    image_desp = []
                    image_env =[] 
                    for vp_id in image_list:
                        if vp_id == None:
                            image_desp.append(None)
                            image_env.append("")
                            continue
                        #image_env.append(self.prompt_manager.make_env_description(obs, vp_id, previous_angle))
                        image_names = vp_id.split("/")
                        viewpoint, image_id = image_names[-2], int(image_names[-1].split(".")[0])
                        #if scan not in pano_desp or viewpoint not in pano_desp[scan]:
                            #self.pano_manager.get_pano_descriptions(pano_desp, scan, viewpoint)
                        image_desp.append(candi_desp[scan][viewpoint][str(image_id)])



                    #################
                    
                    nav_output, tokens = gpt_infer_action_both(nav_input["task_description"], environment_prompts, image_desp, image_list,
                                                   self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})


                    # nav_output, tokens = qwen_infer(nav_input["task_description"], environment_prompts, image_desp, image_list,
                    #                                 self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})
                    
                    json_output= json.loads(nav_output)
                    a_t = self.prompt_manager.parse_json_action(json_output, nav_input["only_options"], t)
                    self.prompt_manager.parse_json_planning(json_output)
                    print('-------------------- Output --------------------')
                    print(nav_output)
                    map_log.append(json_output)

            elif self.args.llm == 'gemini-2.5-pro':
                if len(image_list) > 20:
                        # GPT-4o currently does not support queries with more than 20 images
                    a_t = [0]
                    print('Exceed image limit and stop!')
                else:
                    if scan not in candi_desp:
                        candi_desp[scan] = {}
                    if self.prompt_manager.trajectory[0][-1] not in candi_desp[scan]: 
                        node_list = cand_inputs['map_nodes'][0]
                        if image_list[0] == None:
                            node_list.insert(0, None)

                        new_image_list = []
                        for node in node_list:
                            if node == None:
                                continue
                            new_image_list.append(image_list[node])
                        remove_dup_img = list(dict.fromkeys(new_image_list))
                        tmp_image_desp = []
                        nav_output, tokens = gpt_infer_description(nav_input["img2txt_task"], nav_input["img2txt_prompts"][0], remove_dup_img,
                                                    self.args.llm, 2000, response_format={"type": "json_object"})

                        json_des_output = json.loads(nav_output)
                        tmp_image_desp = list(json_des_output.values())

                        if len(tmp_image_desp) != len(remove_dup_img):
                            print(obs[0]['instr_id'])
                        assert len(tmp_image_desp) == len(remove_dup_img)
                        
                        new_image_dict = dict(zip(remove_dup_img, tmp_image_desp))
                        if image_list[0] == None:
                            tmp_image_desp.insert(0, None)
                        for id, node in enumerate(node_list):
                            if node == None:
                                continue
                            image_names = image_list[node].split("/")
                            viewpoint, image_id = image_names[-2], int(image_names[-1].split(".")[0])
                            if viewpoint not in candi_desp[scan]:
                                candi_desp[scan][viewpoint] = {}
                            if str(image_id) not in candi_desp[scan][viewpoint]:
                                candi_desp[scan][viewpoint][str(image_id)] = new_image_dict[image_list[node]]
                                
                        add_data_to_json(candi_des_path, candi_desp)
                    
                    image_desp = []
                    image_env =[] 
                    for vp_id in image_list:
                        if vp_id == None:
                            image_desp.append(None)
                            image_env.append("")
                            continue
                        #image_env.append(self.prompt_manager.make_env_description(obs, vp_id, previous_angle))
                        image_names = vp_id.split("/")
                        viewpoint, image_id = image_names[-2], int(image_names[-1].split(".")[0])
                        #if scan not in pano_desp or viewpoint not in pano_desp[scan]:
                            #self.pano_manager.get_pano_descriptions(pano_desp, scan, viewpoint)
                        image_desp.append(candi_desp[scan][viewpoint][str(image_id)])



                    #################
                    
                    nav_output = gemini_infer(nav_input["task_description"], 
                                                  environment_prompts, 
                                                  image_desp,
                                                  image_list,
                                                  self.args.llm)
                        # print('-------------------- Output --------------------')
                        # print(nav_output)
                        
                        
                        # JSON file
                    try:
                        if nav_output.startswith('`') or nav_output.endswith('```'):
                        # Attempt to parse as JSON
                            nav_output = nav_output.strip().replace('```json', '').replace('```', '').strip()

                        # print("Parsing JSON Output...")
                        json_output = json.loads(nav_output)
                        # nav_output = json.loads(nav_output)
                        
                        print("-------------------- Parsed JSON Output --------------------")
                        a_t = self.prompt_manager.parse_json_action(json_output=json_output,
                                                        only_options_batch=nav_input["only_options"],
                                                        t=t)
                        self.prompt_manager.parse_json_planning(json_output=json_output)
                    except json.JSONDecodeError:
                        print("Failed to parse JSON, using raw output.")
                        # nav_output = [nav_output]
                        # print(nav_output)
                    
                        nav_output = [nav_output]
                        a_t = self.prompt_manager.parse_action(nav_output=nav_output,
                                                            only_options_batch=nav_input["only_options"],
                                                            t=t)
                        self.prompt_manager.parse_bad_planning(nav_output=nav_output)
            
           


            else:
                raise NotImplemented

            for i in range(batch_size):
                traj[i]['a_t'][t] = a_t[i]

            # Determine stop actions
            a_t_stop = [a_t_i == 0 for a_t_i in a_t]

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i]:
                    cpu_a_t.append(-1)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(a_t[i] - 1)

            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = self.env._get_obs()
            

            previous_angle = [{'heading': ob['heading'],
                               'elevation': ob['elevation']} for ob in obs]

            # we only implement batch_size=1
            if a_t[0] == 0:
                break

            self.prompt_manager.make_history(a_t, nav_input, t)
            traj[i]['map_log'] = map_log
        return traj
