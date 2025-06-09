import requests
import json

from map import generate_paths


HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"

NUM_SEQUENCES = 6

resp = requests.post(f"{BASE_URL}/testing/reset-database")
assert resp.status_code == 200, f'{resp.status_code} {resp.text}'

all_gps = generate_paths(NUM_SEQUENCES)

for i in range(NUM_SEQUENCES):
    resp = requests.post(f'{BASE_URL}/sequences/', json={'name': f'seq{i}', 'duration': 10 + i, 'gps': json.dumps(all_gps[i])})
    assert resp.status_code == 200, f'{resp.status_code} {resp.text}'

    resp = requests.post(f'{BASE_URL}/sequence_uncertainties/', json={
        'avg_uncertainty': 20 + i,
        'max_uncertainty': 20 + i,
        'uncertainty_per_frame': '[1,3,2]',
        'sequence_name': f'seq{i}'
    })
    assert resp.status_code == 200, f'{resp.status_code} {resp.text}'

resp = requests.post(f"{BASE_URL}/datasets/", json={'name': 'ds1', 'creator': 'user', 'train_sequence_names': [f'seq{i}' for i in range(0, 2)], 'val_sequence_names': ['seq4']})
assert resp.status_code == 200, f'{resp.status_code} {resp.text}'

resp = requests.post(f"{BASE_URL}/datasets/", json={'name': 'ds2', 'creator': 'user', 'train_sequence_names': [f'seq{i}' for i in range(2, 4)], 'val_sequence_names': ['seq4', 'seq1']})
assert resp.status_code == 200, f'{resp.status_code} {resp.text}'

resp = requests.post(f"{BASE_URL}/training_runs/", json={'model_pt_path': 'model.pt', 'started_by': 'user', 'dataset_id': 1})
assert resp.status_code == 200, f'{resp.status_code} {resp.text}'
