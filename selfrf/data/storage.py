from abc import abstractmethod
import io
from pathlib import Path
from typing import Protocol, Union

import minio
from minio import Minio


class StorageBackend(Protocol):
    """Abstract base class defining storage backend interface.

    This class serves as a protocol for implementing storage backends,
    defining the required interface for data storage operations.

    Attributes:
        base_path (Path): Base directory path for storage operations

    Methods:
        put_object(data: Union[bytes, io.BytesIO], path: str) -> None:
            Store binary data at specified path
        create_dir(path: str) -> None: 
            Create directory at path if backend supports it
        exists(path: str) -> bool:
            Check if object exists at specified path
        set_base_path(base_path: Union[str, Path]) -> None:
            Set base directory path for storage operations
    """
    base_path: Path

    @abstractmethod
    def put_object(self, data: Union[bytes, io.BytesIO], path: str) -> None:
        """Store object data at path"""
        raise NotImplementedError

    def create_dir(self, path: str) -> None:
        """Optional directory creation - default no-op"""
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if object exists at path"""
        raise NotImplementedError

    def set_base_path(self, base_path: Union[str, Path]) -> None:
        """
        Sets the base path for storage operations.

        Args:
            base_path (Union[str, Path]): The base directory path to be used for storage operations.
                                         Can be provided as a string or Path object.

        """
        self.base_path = Path(base_path)


class MinioBackend(StorageBackend):
    """A storage backend implementation for MinIO object storage.

    This class provides methods to interact with MinIO storage service,
    including uploading objects and checking their existence in a specified bucket.

        client (Minio): Initialized MinIO client object
        bucket (str): Name of the bucket to use for storage
        base_path (Union[str, Path], optional): Base path prefix for all operations. Defaults to "".

    Attributes:
        client (Minio): MinIO client instance
        bucket (str): Name of the bucket being used
        base_path (Path): Base path prefix for all operations
    """

    def __init__(self, client: Minio, bucket: str, base_path: Union[str, Path] = ""):
        self.client: Minio = client
        self.bucket = bucket
        self.set_base_path(base_path)

        # Ensure bucket exists
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def put_object(self, data: Union[bytes, io.BytesIO], path: Path) -> None:
        """Upload data to Minio bucket

        Args:
            data: Bytes or BytesIO object to upload
            path: Object path/key in bucket
        """
        # Convert bytes to BytesIO if needed
        if isinstance(data, bytes):
            data = io.BytesIO(data)

        # Get length and reset position
        data.seek(0, io.SEEK_END)
        length = data.tell()
        data.seek(0)

        path = self._normalize_path(self.base_path / path)
        self.client.put_object(
            bucket_name=self.bucket,
            object_name=path,
            data=data,
            length=length
        )

    def exists(self, path: Path) -> bool:
        try:
            path = self._normalize_path(self.base_path / path)
            self.client.stat_object(self.bucket, path)
            return True
        except minio.error.S3Error:
            return False

    def _normalize_path(self, path: Union[str, Path]) -> str:
        """Convert Windows or Unix path to S3-style object key"""
        return str(self.base_path / path).replace('\\', '/')


class FilesystemBackend(StorageBackend):
    """A storage backend implementation that uses the local filesystem.

    This class provides methods to interact with the local filesystem for storing and managing data.
    It implements the StorageBackend interface by providing concrete implementations for storing
    objects, creating directories, and checking existence of paths.

    Args:
        base_path (Union[str, Path]): The base directory path for all storage operations. Defaults to empty string.

    Attributes:
        base_path (Path): The resolved base path used for all storage operations.
    """

    def __init__(self, base_path: Union[str, Path] = ""):
        self.set_base_path(base_path)

    def put_object(self, data: Union[bytes, io.BytesIO], path: Path) -> None:
        """Store binary data to a file at the specified path.

        This method creates necessary parent directories and writes binary data to a file.

        Args:
            data (Union[bytes, io.BytesIO]): Binary data to store, either as bytes or BytesIO object
            path (Path): Relative path where to store the data
        """
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, io.BytesIO):
            data = data.getvalue()
        full_path.write_bytes(data)

    def create_dir(self, path: Path) -> None:
        """Create a directory at the specified path.

        This method creates a directory at the given path relative to the base path. 
        If parent directories don't exist, they will be created as well.

        Args:
            path (Path): The relative path where the directory should be created.
        """
        full_path = self.base_path / path
        full_path.mkdir(parents=True, exist_ok=True)

    def exists(self, path: Path) -> bool:
        """
        Check if a given path exists relative to the base path.

        Args:
            path (Path): The relative path to check for existence.

        Returns:
            bool: True if the path exists, False otherwise.
        """
        return (self.base_path / path).exists()
