import io
from typing import List
from minio import Minio
from src.config import settings

class MinioService:
    def __init__(self):
        self.client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            if not self.client.bucket_exists(settings.MINIO_BUCKET):
                self.client.make_bucket(settings.MINIO_BUCKET)
                print(f"Bucket '{settings.MINIO_BUCKET}' created successfully.")
        except Exception as e:
            print(f"Warning: Could not connect or create MinIO bucket: {e}")

    def upload_file(self, object_name: str, data: io.BytesIO, length: int, content_type: str = "application/octet-stream") -> str:
        """
        Uploads a file stream to MinIO.
        """
        self.client.put_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=object_name,
            data=data,
            length=length,
            content_type=content_type
        )
        return object_name

    def download_file(self, object_name: str) -> bytes:
        """
        Downloads a file from MinIO and returns it as bytes.
        """
        response = None
        try:
            response = self.client.get_object(
                bucket_name=settings.MINIO_BUCKET,
                object_name=object_name
            )
            return response.read()
        finally:
            if response:
                response.close()
                response.release_conn()

    def list_documents(self) -> List[dict]:
        """
        Lists metadata of all documents in the MinIO bucket.
        """
        objects = self.client.list_objects(settings.MINIO_BUCKET, recursive=True)
        return [
            {
                "name": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified.isoformat() if obj.last_modified else None
            }
            for obj in objects
        ]
