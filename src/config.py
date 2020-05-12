"""Global config: defaults < environment variable"""
import os
from multiprocessing import cpu_count


CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')
CUDA_VISIBLE_DEVICES = [] if CUDA_VISIBLE_DEVICES is None else CUDA_VISIBLE_DEVICES.split(',')
DEVICE0 = f'cuda:{CUDA_VISIBLE_DEVICES[0]}' if len(CUDA_VISIBLE_DEVICES) > 0 else 'cpu'
N_GPUS = int(os.environ.get('N_GPUS', len(CUDA_VISIBLE_DEVICES)))
N_CPUS = int(os.environ.get('N_CPUS', min(max(cpu_count() - 1, 1), 4)))
IS_TEST = bool(int(os.environ.get('TEST', '1')))


print(f'CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}')
print(f'DEVICE0: {DEVICE0}')
print(f'N_GPUS: {N_GPUS}')
print(f'N_CPUS: {N_CPUS}')
print(f'IS_TEST: {IS_TEST}')
