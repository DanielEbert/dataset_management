import requests

# The `api_client` fixture from conftest.py provides the base URL
# and ensures the server is running and the database is clean for each test.

# --- Sequence Tests ---

def test_create_and_read_sequence(api_client):
    """Test creating a sequence and then reading it back."""
    payload = {"name": "Test Sequence 01", "duration": 120.5}
    
    # Create
    create_response = requests.post(f"{api_client}/sequences/", json=payload)
    assert create_response.status_code == 200
    data = create_response.json()
    assert data["name"] == payload["name"]
    assert data["duration"] == payload["duration"]
    assert "id" in data
    seq_id = data["id"]

    # Read
    read_response = requests.get(f"{api_client}/sequences/{seq_id}")
    assert read_response.status_code == 200
    read_data = read_response.json()
    assert read_data == data

def test_read_all_sequences(api_client):
    """Test reading a list of all sequences."""
    # Initial state should be empty
    response_empty = requests.get(f"{api_client}/sequences/")
    assert response_empty.status_code == 200
    assert response_empty.json() == []

    # Add two sequences
    requests.post(f"{api_client}/sequences/", json={"name": "Seq A", "duration": 10})
    requests.post(f"{api_client}/sequences/", json={"name": "Seq B", "duration": 20})

    # Read list again
    response_full = requests.get(f"{api_client}/sequences/")
    assert response_full.status_code == 200
    data = response_full.json()
    assert len(data) == 2
    assert data[0]["name"] == "Seq A"
    assert data[1]["name"] == "Seq B"

def test_read_nonexistent_sequence(api_client):
    """Test that reading a non-existent sequence returns 404."""
    response = requests.get(f"{api_client}/sequences/9999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Sequence not found"

def test_update_sequence(api_client):
    """Test updating an existing sequence."""
    payload = {"name": "Original Name", "duration": 100.0}
    create_response = requests.post(f"{api_client}/sequences/", json=payload)
    seq_id = create_response.json()["id"]

    # Update
    update_payload = {"name": "Updated Name"}
    update_response = requests.patch(f"{api_client}/sequences/{seq_id}", json=update_payload)
    assert update_response.status_code == 200
    updated_data = update_response.json()
    assert updated_data["name"] == "Updated Name"
    assert updated_data["duration"] == 100.0  # Duration should not change

    # Verify by reading again
    read_response = requests.get(f"{api_client}/sequences/{seq_id}")
    assert read_response.json()["name"] == "Updated Name"

def test_create_duplicate_sequence_fails(api_client):
    """Test that creating a sequence with a duplicate name fails."""
    payload = {"name": "Unique Name", "duration": 50.0}
    requests.post(f"{api_client}/sequences/", json=payload)
    
    # Try to create again with the same name
    response = requests.post(f"{api_client}/sequences/", json=payload)
    assert response.status_code == 500  # Database unique constraint error

# --- Sequence Uncertainty Tests ---

def test_create_uncertainty_for_nonexistent_sequence(api_client):
    """Test creating an uncertainty for a sequence that doesn't exist."""
    payload = {
        "sequence_name": "Non-Existent Sequence",
        "avg_uncertainty": 0.5,
        "max_uncertainty": 0.9,
        "uncertainty_per_frame": "[0.1, 0.2, 0.9]"
    }
    response = requests.post(f"{api_client}/sequence_uncertainties/", json=payload)
    assert response.status_code == 404
    assert "doesnt exist" in response.json()["detail"]

def test_create_and_read_uncertainty(api_client):
    """Test creating and reading a sequence uncertainty measurement."""
    seq_name = "Sequence for Uncertainty"
    requests.post(f"{api_client}/sequences/", json={"name": seq_name, "duration": 3.0})

    unc_payload = {
        "sequence_name": seq_name,
        "avg_uncertainty": 0.5,
        "max_uncertainty": 0.9,
        "uncertainty_per_frame": "[0.1, 0.2, 0.9]"
    }
    create_response = requests.post(f"{api_client}/sequence_uncertainties/", json=unc_payload)
    assert create_response.status_code == 200
    data = create_response.json()
    assert data["avg_uncertainty"] == unc_payload["avg_uncertainty"]
    unc_id = data["id"]
    
    read_response = requests.get(f"{api_client}/sequence_uncertainties/{unc_id}")
    assert read_response.status_code == 200
    assert read_response.json()["id"] == unc_id

# --- Dataset Tests ---

def test_create_and_read_dataset(api_client):
    """Test creating a dataset and reading it back."""
    payload = {"name": "My Awesome Dataset", "creator": "Pytest Runner", 'sequence_names': []}
    create_response = requests.post(f"{api_client}/datasets/", json=payload)
    assert create_response.status_code == 200
    data = create_response.json()
    assert data["name"] == payload["name"]
    assert data["creator"] == payload["creator"]
    dataset_id = data["id"]

    read_response = requests.get(f"{api_client}/datasets/{dataset_id}")
    assert read_response.status_code == 200
    assert read_response.json() == data

def test_read_nonexistent_dataset(api_client):
    """Test that reading a non-existent dataset returns 404."""
    response = requests.get(f"{api_client}/datasets/9999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset not found"
