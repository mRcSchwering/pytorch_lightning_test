"""Global config: defaults < config.json < environment variable"""
import os
import json
from multiprocessing import cpu_count
from torch import device, cuda

CONFIG_FILE = 'config.json'
_config = None
if os.path.isfile(CONFIG_FILE):
    with open(CONFIG_FILE) as inf:
        _config = json.load(inf)


def _get(name: str, default: str) -> str:
    if _config is not None:
        val = _config.get(name, default)
    val = os.environ.get(name, default)
    return str(val)


TEST = bool(int(_get('TEST', '1')))
DEVICE = _get('DEVICE', device('cuda' if cuda.is_available() else 'cpu'))
WORKERS = int(_get('WORKERS', min(max(cpu_count() - 1, 1), 4)))


print(f'TEST: {TEST}')
print(f'DEVICE: {DEVICE}')
print(f'WORKERS: {WORKERS}')