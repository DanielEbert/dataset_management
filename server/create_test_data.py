import requests
import json

from map import generate_paths


HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"

resp = requests.post(f"{BASE_URL}/testing/reset-database")
assert resp.status_code == 200, f'{resp.status_code} {resp.text}'

all_gps = generate_paths(5)

for i in range(5):
    resp = requests.post(f'{BASE_URL}/sequences/', json={'name': f'seq{i}', 'duration': 10 + i, 'gps': json.dumps(all_gps[i])})
    assert resp.status_code == 200, f'{resp.status_code} {resp.text}'

    resp = requests.post(f'{BASE_URL}/sequence_uncertainties/', json={
        'avg_uncertainty': 20 + i,
        'max_uncertainty': 20 + i,
        'uncertainty_per_frame': '[1,3,2]',
        'sequence_name': f'seq{i}'
    })
    assert resp.status_code == 200, f'{resp.status_code} {resp.text}'

resp = requests.post(f"{BASE_URL}/datasets/", json={'name': 'ds1', 'creator': 'user', 'sequence_names': [f'seq{i}' for i in range(0, 3)]})
assert resp.status_code == 200, f'{resp.status_code} {resp.text}'

resp = requests.post(f"{BASE_URL}/datasets/", json={'name': 'ds2', 'creator': 'user', 'sequence_names': [f'seq{i}' for i in range(2, 5)]})
assert resp.status_code == 200, f'{resp.status_code} {resp.text}'

resp = requests.post(f"{BASE_URL}/training_runs/", json={'model_pt_path': 'model.pt', 'creator': 'user', 'dataset_id': 1})
assert resp.status_code == 200, f'{resp.status_code} {resp.text}'
