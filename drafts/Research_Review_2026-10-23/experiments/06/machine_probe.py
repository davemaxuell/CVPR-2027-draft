import importlib.util, importlib.metadata, json, platform, sys
out={'executable':sys.executable,'python':platform.python_version(),'platform':platform.platform(),'packages':{}}
for name in ['torch','torchvision','transformers','accelerate','peft','diffusers','timm','habitat','habitat_sim','numpy','psutil']:
    try:
        spec=importlib.util.find_spec(name)
        info={'available':spec is not None, 'origin':getattr(spec,'origin',None)}
        try: info['version']=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    except Exception as e: info={'error':repr(e)}
    out['packages'][name]=info
try:
    import torch
    out['torch_runtime']={'version':torch.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count()}
except Exception as e: out['torch_runtime']={'error':repr(e)}
print(json.dumps(out,indent=2))
