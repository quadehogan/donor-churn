import os
import pickle
import json
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

CONTAINER_NAME = "ml-artifacts"


def _get_container():
    client = BlobServiceClient.from_connection_string(
        os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )
    return client.get_container_client(CONTAINER_NAME)


def upload_artifact(local_path: str, blob_name: str):
    container = _get_container()
    with open(local_path, "rb") as f:
        container.upload_blob(blob_name, f, overwrite=True)
    print(f"Uploaded → {blob_name}")


def download_artifact(blob_name: str, local_path: str):
    container = _get_container()
    blob = container.download_blob(blob_name)
    with open(local_path, "wb") as f:
        f.write(blob.readall())
    print(f"Downloaded ← {blob_name}")


def load_model_from_blob(blob_name: str = "churn/churn_model.pkl"):
    container = _get_container()
    blob = container.download_blob(blob_name)
    return pickle.loads(blob.readall())


def load_json_from_blob(blob_name: str) -> dict:
    container = _get_container()
    blob = container.download_blob(blob_name)
    return json.loads(blob.readall())
