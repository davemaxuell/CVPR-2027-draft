
AVAILABLE_TASKS = {
    "R2R": "R2RDataset",
    "REVERIE": "REVERIEDataset",
    "CVDN": "CVDNDataset",
    "RXR": "RXRDataset",
    "OBJECTNAV": "ObjectNavDataset"
}

def get_data(**kwargs):
    task_name = kwargs['task']
    if task_name not in AVAILABLE_TASKS:
        raise ValueError(f"Task {task_name} not found in available tasks.")
    
    task_class_name = AVAILABLE_TASKS[task_name]
    
    module_path = f"tasks.{task_name.lower()}.{task_class_name}"

    try:
        module_name, class_name = module_path.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        return cls(**kwargs)
    except Exception as e:
        kwargs['logger'].error(f"Failed to import {class_name} from {module_name}: {e}")
        raise
