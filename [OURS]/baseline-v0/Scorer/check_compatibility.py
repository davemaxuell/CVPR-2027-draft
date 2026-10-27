"""Architecture regressions plus pinned upstream CPU integration, without weights.

Run after check_design.py. VAE/encoder and policy hooks use explicitly labeled
test doubles; DINO predictor, CDiT-S/2 and diffusion sampler use actual source.
"""
import copy
import gc
import hashlib
from importlib.metadata import version
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
LOCAL_DEPS = HERE / '.test_dependencies'
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import torch
from torch import nn
from torch.nn import functional as F
from interfaces import (VisualContract, ActionMap, pool_dino_patches, extract_dino_prediction,
                        instruction_states, last_valid_state, PrefillFeatureTap, validate_policy_config)
from scorer import NavigationScorer, START, STOP, choose_action
from runtime import ScorerRuntime
from world_models import (build_dino_predictor, DinoCommandDynamics, DinoRGBEncoder,
                          NWMVisualBridge, load_pinned_source, NWM_MODELS_SHA256,
                          CheckedCDiTDenoiser, cdit_conditioning, build_nwm_diffusion,
                          sample_nwm_latents)

ROOT = HERE.parent
SOURCE = ROOT / 'third_party'
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8-sig'))
DETAILS = {}

def rejects(call, exception=(ValueError, TypeError, RuntimeError)):
    try:
        call()
    except exception:
        return
    raise AssertionError('An incompatible input was silently accepted')

def small_config():
    cfg = copy.deepcopy(CONFIG)
    cfg['architecture'].update(width=64, memory_slots=3, candidate_blocks=1,
                               feedforward_width=128, readout_width=32)
    return cfg

def features(model, batch=2, dtype=torch.float32):
    return dict(instruction=torch.randn(batch,6,model.policy_width,dtype=dtype),
                instruction_valid=torch.tensor([[True]*4+[False]*2]*batch),
                current_patches=torch.randn(batch,model.patch_count,model.visual.feature_width,dtype=dtype),
                policy_context=torch.randn(batch,model.policy_width,dtype=dtype),
                previous_action=torch.full((batch,),START,dtype=torch.long),
                elapsed_fraction=torch.zeros(batch,1,dtype=dtype))

def scorer_matrix():
    observed = []
    for backbone in CONFIG['text_and_context']['hidden_widths']:
        for grid in (14,16):
            cfg = copy.deepcopy(CONFIG)
            cfg['visual_features'].update(raw_grid=grid,input_resolution=grid*14,
                                           raw_patches=grid**2,pooled_patches=(grid//2)**2)
            model = NavigationScorer.from_config(cfg, backbone)
            args = features(model, batch=1)
            ctx = model.observe(**args)
            actions = torch.arange(4)[None]
            predicted = torch.randn(1,4,model.patch_count,384)
            out = model.score(ctx,predicted,actions)
            loss = model.loss(out,torch.tensor([[.8,.2,.1,.05]]),torch.tensor([[1.,0.,0.,0.]]),torch.zeros(1,4),actions)
            loss['total'].backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum().item() for g in grads) > 0
            observed.append(dict(backbone=backbone,grid=grid,patches=model.patch_count,loss=float(loss['total'].detach())))
            del model, ctx, out, loss, grads
            gc.collect()
    DETAILS['scorer_matrix'] = observed

def configuration_and_metadata():
    cfg = small_config()
    model = NavigationScorer.from_config(cfg,'smolvlm2')
    assert (model.width,model.slots,len(model.blocks),model.readout[1].out_features)==(64,3,1,32)
    cfg['loss'].update(terminal_success_bce=0.,immediate_forward_collision_bce=0.)
    model = NavigationScorer.from_config(cfg,'smolvlm2')
    out = model.score(model.observe(**features(model,1)),torch.randn(1,4,49,384),torch.arange(4)[None])
    losses = model.loss(out,torch.zeros(1,4),torch.zeros(1,4),torch.zeros(1,4),torch.arange(4)[None])
    torch.testing.assert_close(losses['total'],losses['value'])
    bad=copy.deepcopy(cfg); bad['visual_features']['raw_patches']=256
    rejects(lambda: NavigationScorer.from_config(bad,'smolvlm2'))
    bad=copy.deepcopy(cfg); bad['prediction_horizon_actions']=2
    rejects(lambda: NavigationScorer.from_config(bad,'smolvlm2'))
    bad=copy.deepcopy(cfg); bad['selection']['override_margin']=-.1
    rejects(lambda: NavigationScorer.from_config(bad,'smolvlm2'))
    pins=json.loads((ROOT/'manifest/source_pins.json').read_text(encoding='utf-8-sig'))
    records = []
    for key, metadata in zip(('smolvlm2','onevision','qwen3vl'),pins['backbone_metadata']):
        width=CONFIG['text_and_context']['hidden_widths'][key]
        validate_policy_config(key,metadata['config'],width)
        rejects(lambda: validate_policy_config(key,metadata['config'],width+1))
        records.append(dict(backbone=key,id=metadata['id'],revision=metadata['revision'],text_width=width))
    DETAILS['checked_policy_metadata_not_weights']=records

def masks_and_precision():
    model=NavigationScorer.from_config(small_config(),'smolvlm2').eval()
    for dtype in (torch.float16,torch.bfloat16):
        args=features(model,dtype=dtype)
        args['instruction'][~args['instruction_valid']]=float('nan')
        context=model.observe(**args)
        out=model.score(context,torch.randn(2,4,49,384,dtype=dtype),torch.arange(4).expand(2,-1))
        assert all(x.dtype==torch.float32 and torch.isfinite(x).all() for x in out.values())
        out['utility'].sum().backward()
        model.zero_grad(set_to_none=True)
    args=features(model)
    args['instruction'][0,0,0]=float('nan')
    rejects(lambda:model.observe(**args))
    args=features(model);args['previous_action']=torch.zeros(2)
    rejects(lambda:model.observe(**args))
    args=features(model);args['current_patches']=torch.randn(2,64,384)
    rejects(lambda:model.observe(**args))
    args=features(model);args['instruction_valid'].zero_()
    rejects(lambda:model.observe(**args))

def memory_and_gradients():
    model=NavigationScorer.from_config(small_config(),'smolvlm2').eval()
    args=features(model)
    memory=torch.randn(2,3,64,requires_grad=True)
    context=model.observe(**args,memory=memory,reset_mask=torch.tensor([True,False]))
    fresh=model.observe(**args)
    torch.testing.assert_close(context.memory[0],fresh.memory[0])
    context.memory.sum().backward()
    assert memory.grad[0].abs().sum()==0 and memory.grad[1].abs().sum()>0
    future=torch.randn(2,4,49,384,requires_grad=True)
    context=model.observe(**args)
    model.score(context,future,torch.arange(4).expand(2,-1))['utility'].sum().backward()
    assert future.grad is None
    rejects(lambda:model.score(context,future,torch.arange(4).expand(2,-1),future_grad=True))
    model.zero_grad(set_to_none=True);model.requires_grad_(False)
    future=future.detach().requires_grad_(True)
    model.score(context,future,torch.arange(4).expand(2,-1),future_grad=True)['utility'].sum().backward()
    assert future.grad[:,:3].abs().sum()>0 and future.grad[:,3].abs().sum()==0
    assert torch.isfinite(future.grad).all() and all(p.grad is None for p in model.parameters())
    before=context.memory.detach().clone()
    broken=future.detach().clone();broken[:,3]=float('nan')
    assert torch.isfinite(model.score(context,broken,torch.arange(4).expand(2,-1))['utility']).all()
    torch.testing.assert_close(before,context.memory)

def policy_features():
    hidden=torch.arange(2*6*8,dtype=torch.float32).reshape(2,6,8)
    mask=torch.tensor([[True,True,True,False,False,False],[False,False,True,True,True,True]])
    torch.testing.assert_close(last_valid_state(hidden,mask),torch.stack((hidden[0,2],hidden[1,5])))
    text,valid=instruction_states(hidden.requires_grad_(True),mask)
    assert text.shape==(2,4,8) and valid.sum().item()==7 and not text.requires_grad
    torch.testing.assert_close(text[0,:3],hidden[0,:3]);torch.testing.assert_close(text[1],hidden[1,2:])
    rejects(lambda:last_valid_state(hidden,mask[:,:5]))
    norm=nn.LayerNorm(8).eval()
    tap=PrefillFeatureTap(norm,8)
    with tap.capture(): expected=norm(hidden)
    torch.testing.assert_close(tap.hidden,expected)
    assert not tap.hidden.requires_grad and len(norm._forward_hooks)==0
    def many_calls():
        with tap.capture():norm(hidden);norm(hidden[:,:1])
    rejects(many_calls)
    assert tap.hidden is None and len(norm._forward_hooks)==0
    def no_calls():
        with tap.capture():pass
    rejects(no_calls)
    def failed_prefill():
        with tap.capture():
            norm(hidden)
            raise RuntimeError('Simulated failure after the final norm')
    rejects(failed_prefill)
    assert tap.hidden is None and len(norm._forward_hooks)==0

def prediction_layouts():
    contract=VisualContract()
    raw=torch.randn(2,3,196,384)
    expected=pool_dino_patches(raw[:,-1])
    combined=torch.cat((raw,torch.full((2,3,196,20),1.e6)),dim=-1)
    token=torch.cat((raw,torch.full((2,3,2,384),1.e6)),dim=-2)
    torch.testing.assert_close(extract_dino_prediction(combined,contract,layout='feature_concat',condition_width=20),expected)
    torch.testing.assert_close(extract_dino_prediction(token,contract,layout='token_concat'),expected)
    torch.testing.assert_close(extract_dino_prediction(raw,contract,layout='visual_only'),expected)
    rejects(lambda:extract_dino_prediction(combined,contract,layout='visual_only'))
    rejects(lambda:pool_dino_patches(torch.randn(2,256,384)))
    assert pool_dino_patches(torch.randn(2,4,256,384),VisualContract(raw_grid=16)).shape==(2,4,64,384)

def fallback_and_lifecycle():
    acts=torch.arange(4).expand(2,-1)
    choice=choose_action(torch.tensor([[.1,.8,.2,.3],[float('nan'),.8,.2,.3]]),acts,torch.tensor([0,2]),.1)
    assert choice.actions.tolist()==[1,2] and choice.reasons==['override','nonfinite_scores']
    assert choose_action(None,acts,torch.tensor([1,2]),0.).actions.tolist()==[1,2]
    mapping=ActionMap((9,4,7,2))
    torch.testing.assert_close(mapping.from_simulator(mapping.to_simulator(acts)),acts)
    rejects(lambda:mapping.from_simulator(torch.tensor([0])))
    model=NavigationScorer.from_config(small_config(),'smolvlm2')
    runtime=ScorerRuntime(model)
    args=features(model,1)
    runtime.observe(0,**args)
    def forbidden(_):raise AssertionError('Disabled overrides must not call the world model')
    assert runtime.decide(0,forbidden)==0 and runtime.events[-1]['reason']=='disabled'
    runtime.margin=.1
    assert runtime.decide(1,lambda _:torch.zeros(1,3,64,384))==1
    assert runtime.events[-1]['prediction_error']
    assert runtime.decide(2,lambda _:torch.full((1,3,49,384),float('nan')))==2
    assert runtime.decide(0,lambda _:torch.zeros(1,3,49,384,dtype=torch.long))==0
    before=runtime.context.memory.clone()
    runtime.decide(0,lambda _:torch.randn(1,3,49,384))
    torch.testing.assert_close(before,runtime.context.memory)
    rejects(lambda:runtime.observe(0,**args))
    rejects(lambda:runtime.observe(1,**args))
    runtime.commit(1)
    rejects(lambda:runtime.observe(1,**args))
    args['previous_action']=torch.tensor([1]);runtime.observe(1,**args)
    runtime.commit(STOP)
    rejects(lambda:runtime.observe(2,**args))
    runtime.reset();args['previous_action']=torch.tensor([START]);runtime.observe(0,**args)

def upstream_dino():
    source=SOURCE/'dino_wm/models/vit.py'
    predictor=build_dino_predictor(source)
    dynamics=DinoCommandDynamics(predictor).eval()
    masks={k:v for k,v in predictor.named_buffers() if k.endswith('bias')}
    assert len(masks)==6 and all(v.device.type=='cpu' for v in masks.values())
    assert not (masks.keys() & predictor.state_dict().keys())
    raw=torch.randn(1,3,196,384)
    with torch.no_grad():
        next_raw=dynamics(raw,torch.tensor([[0,1]]),torch.tensor([2]))
        initial=dynamics(raw[:,:1],torch.empty(1,0,dtype=torch.long),torch.tensor([0]))
    assert next_raw.shape==initial.shape==(1,196,384) and torch.isfinite(next_raw).all()
    model=NavigationScorer.from_config(small_config(),'smolvlm2').eval()
    runtime=ScorerRuntime(model,margin=.1)
    args=features(model,1);args['current_patches']=pool_dino_patches(raw[:,-1]);runtime.observe(0,**args)
    def predict(ids):
        return torch.stack([pool_dino_patches(dynamics(raw,torch.tensor([[0,1]]),a)) for a in ids.unbind(1)],dim=1)
    assert runtime.decide(0,predict) in range(4) and runtime.events[-1]['prediction_error'] is None
    # Actual six-layer predictor gradient path; fresh CE embedding receives a gradient.
    dynamics.train();actual=dynamics(raw,torch.tensor([[0,1]]),torch.tensor([2]))
    F.mse_loss(actual,torch.randn_like(actual)).backward()
    assert dynamics.command_embedding.weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in dynamics.parameters() if p.grad is not None)
    dynamics.load_transferred_predictor(predictor.state_dict())
    incompatible=dict(predictor.state_dict());incompatible['pos_embedding']=torch.randn(1,768,404)
    rejects(lambda:dynamics.load_transferred_predictor(incompatible))
    rejects(lambda:dynamics(raw,torch.tensor([[0,1]]),torch.tensor([STOP])))
    DETAILS['dino_predictor']={'parameters':sum(p.numel() for p in predictor.parameters()),'depth':6,'history':3,'grid':14,'width':404,'weights':'random; no released checkpoint','portable_mask':'nonpersistent registered buffer; upstream file unchanged'}

class ToyDino(nn.Module):
    """Shape/preprocessing/gradient test double, NOT actual DINOv2."""
    patch_size=14
    num_features=384
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.ones(1));self.last=None
    def forward_features(self,x):
        self.last=x.detach()
        small=F.avg_pool2d(x,14).mean(1).flatten(1)
        return {'x_norm_patchtokens':small[:,:,None].expand(-1,-1,384)*self.weight}

class ToyVAE(nn.Module):
    """Scale/shape/gradient test double, NOT the pretrained VAE."""
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.ones(1));self.config=SimpleNamespace(latent_channels=4);self.last_decode=None
    def encode(self,x):
        latent=F.avg_pool2d(x.mean(1,keepdim=True),8).expand(-1,4,-1,-1)*self.weight
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator=None:latent,mode=lambda:latent))
    def decode(self,x):
        self.last_decode=x.detach()
        return SimpleNamespace(sample=F.interpolate(x[:,:3],scale_factor=8,mode='nearest')*self.weight)

def visual_bridge_double():
    encoder=ToyDino();dino=DinoRGBEncoder(encoder);vae=ToyVAE();bridge=NWMVisualBridge(vae,dino)
    bridge.train()
    assert not encoder.training and not vae.training
    real=torch.full((1,4,3,224,224),.75)
    latent=bridge.encode_context(real)
    torch.testing.assert_close(latent,torch.full((1,4,4,28,28),.5*.18215))
    out=bridge(latent[:,0])
    assert out.shape==(1,49,384)
    torch.testing.assert_close(out,torch.full_like(out,.5))
    torch.testing.assert_close(vae.last_decode,torch.full((1,4,28,28),.5))
    assert encoder.last.shape==(1,3,196,196)
    raw=torch.full((1,4,28,28),.2*.18215,requires_grad=True)
    bridge(raw,input_grad=True).sum().backward()
    assert raw.grad.abs().sum()>0 and torch.isfinite(raw.grad).all()
    assert all(p.grad is None and not p.requires_grad for p in bridge.parameters())
    assert not bridge(raw).requires_grad
    rejects(lambda:bridge(torch.zeros(1,8,28,28)))
    rejects(lambda:dino(torch.full((1,3,224,224),2.)))

def upstream_nwm():
    upstream=SOURCE/'nwm'
    module=load_pinned_source(upstream/'models.py',NWM_MODELS_SHA256)
    model=module.CDiT_S_2(input_size=28,context_size=4,in_channels=4).eval()
    denoiser=CheckedCDiTDenoiser(model).eval()
    context=torch.randn(1,4,4,28,28)
    table=torch.tensor([[.1,0.,0.],[0.,0.,.1],[0.,0.,-.1]]) # synthetic declared commands
    condition=cdit_conditioning(torch.tensor([0]),table,context_latents=context,model=model)
    with torch.no_grad():out=denoiser(torch.randn(1,4,28,28),torch.tensor([500]),**condition)
    assert out.shape==(1,8,28,28) and torch.isfinite(out).all()
    rejects(lambda:cdit_conditioning(torch.tensor([0]),table,context_latents=context[:,:3],model=model))
    rejects(lambda:cdit_conditioning(torch.tensor([STOP]),table,context_latents=context,model=model))
    rejects(lambda:cdit_conditioning(torch.tensor([0]),table,context_latents=context,model=model,steps=float('nan')))
    diffusion=build_nwm_diffusion(upstream/'diffusion',sampling_steps=2) # interface check ONLY
    before=torch.random.get_rng_state().clone()
    sample=sample_nwm_latents(denoiser,diffusion,condition,seed=23)
    assert sample.shape==(1,4,28,28) and torch.isfinite(sample).all()
    assert torch.equal(before,torch.random.get_rng_state())
    repeated=sample_nwm_latents(denoiser,diffusion,condition,seed=23)
    torch.testing.assert_close(sample,repeated,rtol=0,atol=0)
    bridge=NWMVisualBridge(ToyVAE(),DinoRGBEncoder(ToyDino()))
    assert bridge(sample).shape==(1,49,384)
    rejects(lambda:bridge(out))
    # Actual CDiT source with BF16 weights exercises its FP32 timestep embedding boundary.
    denoiser.to(dtype=torch.bfloat16)
    with torch.no_grad():mixed=denoiser(torch.randn(1,4,28,28),torch.tensor([500]),**condition)
    assert mixed.shape==(1,8,28,28) and torch.isfinite(mixed).all()
    DETAILS['nwm']={'architecture':'CDiT-S/2','parameters':sum(p.numel() for p in model.parameters()),'context':4,'image':224,'latent':[4,28,28],'denoiser_output':[8,28,28],'sampler_steps_for_interface_test_only':2,'weights':'random; no released checkpoint','vae_and_dino':'test doubles only','bf16_cpu_forward':True}

def main():
    torch.set_num_threads(2);torch.manual_seed(23)
    tests=[scorer_matrix,configuration_and_metadata,masks_and_precision,memory_and_gradients,
           policy_features,prediction_layouts,fallback_and_lifecycle,upstream_dino,
           visual_bridge_double,upstream_nwm]
    records=[]
    for test in tests:
        started=time.perf_counter()
        try:
            test();row={'name':test.__name__,'passed':True}
        except Exception:
            row={'name':test.__name__,'passed':False,'traceback':traceback.format_exc()}
        row['seconds']=round(time.perf_counter()-started,3);records.append(row)
        print(json.dumps(row),flush=True);gc.collect()
    result={'completed_at_utc':datetime.now(timezone.utc).isoformat(),
            'python':platform.python_version(),'torch':torch.__version__,'device':'cpu',
            'dependencies':{name:version(name) for name in ('torchvision','numpy','einops','timm')},
            'cuda_available':torch.cuda.is_available(),'seed':23,
            'all_passed':all(r['passed'] for r in records),'checks':records,'details':DETAILS,
            'scope':'Synthetic features and actual pinned randomly initialized DINO predictor/CDiT/diffusion CPU execution. No pretrained policy/DINO/VAE/world-model weights, CE episodes, training outcomes or GPU profiling.',
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
            'config_sha256':hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest()}
    (HERE/'compatibility_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('All compatibility groups passed:',result['all_passed'],flush=True)
    if not result['all_passed']:raise SystemExit(1)

if __name__=='__main__':main()
