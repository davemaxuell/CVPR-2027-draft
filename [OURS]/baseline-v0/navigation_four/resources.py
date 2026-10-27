import subprocess
import threading
import time
from pathlib import Path
from scorer_features.common import now,read,write
from navigation_four.jobs import run_job


def gpu_process_memory():
    result=subprocess.run(['nvidia-smi','-i','0','--query-compute-apps=pid,used_gpu_memory',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    rows={}
    for line in result.stdout.splitlines():
        fields=[x.strip() for x in line.split(',')]
        if len(fields)==2 and fields[0].isdigit() and fields[1].isdigit():rows[int(fields[0])]=int(fields[1])
    return rows


def monitored_run(job_path):
    job=read(job_path);out=Path(job['output']);stop=threading.Event();samples=[];errors=[];peaks={}
    if (out/'summary.json').exists() and (out/'gpu_memory.json').exists():return run_job(job_path)
    def monitor():
        while not stop.is_set():
            try:
                state=read(out/'status.json') if (out/'status.json').exists() else {}
                pids=state.get('children',[])
                measured=gpu_process_memory() if pids else {}
                selected={str(pid):measured[pid] for pid in pids if pid in measured}
                if selected:
                    samples.append({'utc':now(),'process_mib':selected})
                    for pid,value in selected.items():peaks[pid]=max(peaks.get(pid,0),value)
            except Exception as exc:errors.append(repr(exc))
            stop.wait(2)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:return run_job(job_path)
    finally:
        stop.set();thread.join(timeout=15)
        write(out/'gpu_memory.json',{'status':'sampled' if samples else 'no_samples','completed_utc':now(),
            'device':'physical GPU 0','sampling_interval_seconds':2,'samples':samples,'errors':errors,
            'peak_mib_by_process':peaks,'max_process_peak_mib':max(peaks.values(),default=None),
            'peak_sum_of_worker_processes_mib':max((sum(s['process_mib'].values()) for s in samples),default=None),
            'scope':'nvidia-smi per-process device memory including CUDA/Habitat allocations; sampled peak, not continuous high-water mark; excludes other jobs'})
