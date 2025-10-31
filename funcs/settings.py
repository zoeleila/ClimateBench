import sys
sys.path.append('.')

from pathlib import Path


SCRATCH_DIR = Path('/scratch/globc/garcia/ClimateBench')

RAW_DIR = SCRATCH_DIR / 'rawdata'

GRAPHS_DIR = SCRATCH_DIR / 'graphs'

RUNS_DIR = SCRATCH_DIR / 'runs'

DATASET_DIR = SCRATCH_DIR / 'dataset'

PREDICTIONS_DIR = SCRATCH_DIR / 'predictions'