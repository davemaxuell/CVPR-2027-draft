from collections import deque
from pathlib import Path
import time
import torch
from Scorer.interfaces import pool_dino_patches
from Scorer.scorer import NavigationScorer,choose_action
from policy.online import OnlineInputs
from scorer_features.common import ENCODER_ID,FROZEN,LIVE_TO_CANONICAL,POLICY_ID,native_flags,read,sha
from scorer_features.policy import FrozenPolicyFeatures
from scorer_training.replay import NullFutureScorer
from wm.dino import build_encoder,preprocessing
from wm.frozen import FrozenDinoWorldModel

CANONICAL_TO_LIVE=(1,2,3,0)


class NavigationController:
    def __init__(self,mode='policy',checkpoint=None,world_model=None,margin=None):
        assert mode in ('policy','predicted','null','actual')
        native_flags();torch.backends.mha.set_fastpath_enabled(False)
        self.mode=mode;self.margin=margin;self.extractor=FrozenPolicyFeatures()
        self.policy=self.extractor.policy;self.scorer=None;self.encoder=None;self.dynamics=None
        if mode!='policy':
            assert checkpoint is not None
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert state['variant']==('null' if mode=='null' else 'pooled')
            cls=NullFutureScorer if mode=='null' else NavigationScorer
            self.scorer=cls.from_config(state['config'],'smolvlm2').cuda().eval().requires_grad_(False)
            self.scorer.load_state_dict(state['model'],strict=True)
            self.encoder=build_encoder(assets=FROZEN/'encoder')
            self.transform=preprocessing()
            for f,h in read(FROZEN/'encoder/file_sha256.json').items():assert sha(FROZEN/'encoder'/f)==h
            if mode=='predicted':
                assert world_model in ('dinowm_transfer','dinov2_fresh_dynamics')
                self.dynamics=FrozenDinoWorldModel(FROZEN/world_model,expected_policy_identity=POLICY_ID)
        if margin is not None:assert isinstance(margin,(int,float)) and 0<=margin<float('inf')
        self.reset('',1,initialize_instruction=False)

    @torch.inference_mode()
    def encode(self,image):
        x=self.transform(image)[None].cuda()
        with torch.autocast('cuda',enabled=False):raw=self.encoder.encoder.forward_features(x)['x_norm_patchtokens'].half()
        return raw,pool_dino_patches(raw.float())

    @torch.inference_mode()
    def reset(self,instruction,budget,*,initialize_instruction=True):
        self.instruction=instruction;self.budget=budget;self.online=OnlineInputs(self.policy.inputs,self.policy.device)
        self.memory=None;self.raw_history=deque(maxlen=3);self.committed=[];self.pending=False;self.terminated=False
        if initialize_instruction and self.scorer is not None:
            features=self.extractor.instruction(instruction)
            self.text=features['states'][None].cuda();self.text_valid=features['valid'][None].cuda()

    @torch.inference_mode()
    def observe(self,image,actions):
        assert list(actions)==self.committed and not self.pending and not self.terminated
        t=len(actions);assert t<self.budget
        self.online.append(image)
        batch=self.online(self.instruction,actions)
        native=self.extractor.context(batch)
        self.base_live=native['base_live'];self.base=LIVE_TO_CANONICAL[self.base_live]
        self.native=native
        if self.scorer is not None:
            raw,current=self.encode(image);self.raw_history.append(raw[0])
            previous=4 if not actions else LIVE_TO_CANONICAL[actions[-1]]
            self.context=self.scorer.observe(self.text,self.text_valid,current,native['context'][None].cuda(),
                torch.tensor([previous],device='cuda'),torch.tensor([[t/self.budget]],device='cuda'),memory=self.memory)
            self.memory=self.context.memory
        self.pending=True
        return native

    @torch.inference_mode()
    def futures(self,preview=None):
        t=len(self.committed);assert t>=3
        if self.mode=='predicted':
            history=torch.stack(list(self.raw_history))[None]
            past=torch.tensor([[LIVE_TO_CANONICAL[a] for a in self.committed[-2:]]],device='cuda')
            movement=[]
            for a in range(3):
                self.attempted_predictions+=1
                raw=self.dynamics.predict(history,past,torch.tensor([a],device='cuda'),observations_seen=t+1,encoder_identity=ENCODER_ID)
                movement.append(pool_dino_patches(raw.float())[0])
            movement=torch.stack(movement)[None]
        elif self.mode=='actual':
            assert preview is not None
            images=preview();assert len(images)==3
            movement=torch.stack([self.encode(image)[1][0] for image in images])[None]
        elif self.mode=='null':
            movement=torch.zeros(1,3,49,384,device='cuda')
        else:raise ValueError('Policy-only mode has no candidate futures')
        return torch.cat((movement,self.context.current_patches[:,None]),dim=1)

    @torch.inference_mode()
    def decide(self,preview=None):
        assert self.pending and not self.terminated
        t=len(self.committed);selected=self.base;error=None;scores=None
        reason='policy_only' if self.mode=='policy' else 'disabled' if self.margin is None else 'insufficient_real_context'
        self.attempted_predictions=0
        if self.scorer is not None and self.margin is not None and t>=3:
            before=self.memory.clone()
            try:
                future=self.futures(preview)
                actions=torch.arange(4,device='cuda')[None]
                out=self.scorer.score(self.context,future,actions)
                if all(torch.isfinite(v).all() for v in out.values()):
                    decision=choose_action(out['utility'],actions,torch.tensor([self.base],device='cuda'),self.margin)
                    selected=int(decision.actions[0]);reason=decision.reasons[0]
                    scores={k:v[0].cpu().tolist() for k,v in out.items()}
                else:reason='nonfinite_scores';selected=self.base
                assert torch.equal(before,self.memory),'Candidate evaluation changed real memory'
            except (ValueError,TypeError,RuntimeError) as exc:
                error=f'{type(exc).__name__}: {exc}';reason='prediction_unavailable';selected=self.base
        live=CANONICAL_TO_LIVE[selected]
        event={'t':t,'base_live':self.base_live,'selected_live':live,'base_canonical':self.base,
            'selected_canonical':selected,'reason':reason,'prediction_error':error,'world_model_calls':self.attempted_predictions,
            'raw_action_scores':self.native['raw_logits'].tolist(),'biased_action_scores':self.native['biased_logits'].tolist(),
            'prompt_length':self.native['prompt_length'],'scores':scores}
        return live,event

    def commit(self,live):
        assert self.pending and not self.terminated and live in range(4)
        self.committed.append(live);self.pending=False;self.terminated=live==0
