import os, pickle
import numpy as np
import seaborn as sns

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

DATA_DIR = 'BrainObservatoryData'
MANIFEST_FILE = os.path.join(DATA_DIR, 'manifest.json')
boc = BrainObservatoryCache(manifest_file=MANIFEST_FILE)

experiments = boc.get_ophys_experiments()

for experiment in experiments:
    experiment_id = experiment['id']
    print("Processing Experiment ", experiment_id)
    events = boc.get_ophys_experiment_events(experiment_id)
    data = boc.get_ophys_experiment_data(experiment_id)