from minio import Minio
import os

BUCKET_NAME=os.getenv("MINIO_BUCKET_NAME", "easyread")
MINIO_HOST=os.getenv("MINIO_HOST", "minio")
MINIO_PORT=os.getenv("MINIO_PORT", "9000")
MINIO_ROOT_USER=os.getenv("MINIO_ROOT_USER", "easyread")
MINIO_ROOT_PASSWORD=os.getenv("MINIO_ROOT_PASSWORD", "Us5l21N85KHD")



class StorageDriver:

    def __init__(self):
        self.client: Minio = Minio(
            f"{MINIO_HOST}:{MINIO_PORT}",
            access_key=MINIO_ROOT_USER,
            secret_key=MINIO_ROOT_PASSWORD,
            secure=False
        )

        # Ensure the bucket exists
        if not self.client.bucket_exists(BUCKET_NAME):
            self.client.make_bucket(BUCKET_NAME)

    def upload_file(self, object_name: str, file_path: str, content_type: str = "application/octet-stream"):
        self.client.fput_object(
            BUCKET_NAME,
            object_name,
            file_path,
            content_type=content_type
        )
    
    def get_file_url(self, object_name: str) -> str:
        return self.client.presigned_get_object(BUCKET_NAME, object_name)



