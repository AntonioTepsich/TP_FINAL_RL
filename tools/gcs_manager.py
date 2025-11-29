"""
GCS Manager for PPO Flappy Bird Training
Handles all Google Cloud Storage operations for experiments, checkpoints, and logs.
"""

import os
import json
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError

logger = logging.getLogger(__name__)


class GCSManager:
    """Manages uploads/downloads to Google Cloud Storage for ML experiments."""

    def __init__(
        self,
        bucket_name: str,
        project_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        local_cache_dir: str = "./gcs_cache"
    ):
        """
        Initialize GCS Manager.

        Args:
            bucket_name: Name of the GCS bucket
            project_id: GCP project ID (uses ADC if None)
            experiment_id: Unique experiment identifier (auto-generated if None)
            local_cache_dir: Local directory to cache downloads
        """
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.experiment_id = experiment_id or self._generate_experiment_id()
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GCS client
        try:
            self.client = storage.Client(project=project_id)
            self.bucket = self.client.bucket(bucket_name)
            logger.info(f"Connected to GCS bucket: {bucket_name}")
            logger.info(f"Experiment ID: {self.experiment_id}")
        except Exception as e:
            logger.error(f"Failed to connect to GCS: {e}")
            raise

        # Thread pool for async uploads
        self._upload_threads: List[threading.Thread] = []

    def _generate_experiment_id(self, config: Optional[Dict] = None) -> str:
        """Generate unique experiment ID with timestamp and optional config hash."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if config:
            config_str = json.dumps(config, sort_keys=True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
            return f"exp_{timestamp}_{config_hash}"
        else:
            return f"exp_{timestamp}"

    def get_experiment_path(self, subdir: str = "") -> str:
        """Get GCS path for experiment."""
        base_path = f"experiments/{self.experiment_id}"
        if subdir:
            return f"{base_path}/{subdir}"
        return base_path

    def upload_file(
        self,
        local_path: str,
        gcs_subpath: str,
        async_upload: bool = False,
        compress: bool = False
    ) -> bool:
        """
        Upload a file to GCS.

        Args:
            local_path: Local file path
            gcs_subpath: Path within experiment directory (e.g., "checkpoints/best_model.pt")
            async_upload: If True, upload in background thread
            compress: If True, compress file before upload (not implemented yet)

        Returns:
            True if upload succeeded (or started for async)
        """
        if not os.path.exists(local_path):
            logger.error(f"Local file not found: {local_path}")
            return False

        gcs_path = f"{self.get_experiment_path()}/{gcs_subpath}"

        def _upload():
            try:
                blob = self.bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                file_size = os.path.getsize(local_path) / 1024  # KB
                logger.info(f"Uploaded {gcs_subpath} ({file_size:.1f} KB) to gs://{self.bucket_name}/{gcs_path}")
                return True
            except GoogleCloudError as e:
                logger.error(f"Failed to upload {local_path} to GCS: {e}")
                return False

        if async_upload:
            thread = threading.Thread(target=_upload, daemon=True)
            thread.start()
            self._upload_threads.append(thread)
            return True
        else:
            return _upload()

    def upload_directory(
        self,
        local_dir: str,
        gcs_subpath: str,
        pattern: str = "*",
        async_upload: bool = False
    ) -> int:
        """
        Upload all files in a directory matching pattern.

        Args:
            local_dir: Local directory path
            gcs_subpath: Base path in GCS (e.g., "tensorboard")
            pattern: File pattern to match (e.g., "*.pt", "events.*")
            async_upload: Upload files asynchronously

        Returns:
            Number of files uploaded (or queued for upload)
        """
        local_dir = Path(local_dir)
        if not local_dir.exists():
            logger.warning(f"Directory not found: {local_dir}")
            return 0

        files = list(local_dir.glob(pattern))
        count = 0

        for file_path in files:
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dir)
                gcs_file_path = f"{gcs_subpath}/{relative_path}"
                if self.upload_file(str(file_path), gcs_file_path, async_upload):
                    count += 1

        return count

    def download_file(self, gcs_subpath: str, local_path: str) -> bool:
        """
        Download a file from GCS.

        Args:
            gcs_subpath: Path within experiment directory
            local_path: Local destination path

        Returns:
            True if download succeeded
        """
        gcs_path = f"{self.get_experiment_path()}/{gcs_subpath}"

        try:
            blob = self.bucket.blob(gcs_path)

            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            blob.download_to_filename(local_path)
            logger.info(f"Downloaded gs://{self.bucket_name}/{gcs_path} to {local_path}")
            return True
        except GoogleCloudError as e:
            logger.error(f"Failed to download {gcs_path}: {e}")
            return False

    def file_exists(self, gcs_subpath: str) -> bool:
        """Check if a file exists in GCS."""
        gcs_path = f"{self.get_experiment_path()}/{gcs_subpath}"
        blob = self.bucket.blob(gcs_path)
        return blob.exists()

    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save experiment configuration to GCS.

        Args:
            config: Configuration dictionary

        Returns:
            True if save succeeded
        """
        config_path = self.local_cache_dir / "config.json"

        full_config = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": config,
        }

        # Save locally
        with open(config_path, 'w') as f:
            json.dump(full_config, f, indent=2)

        # Upload to GCS
        return self.upload_file(str(config_path), "config.json", async_upload=False)

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "final_metrics.json") -> bool:
        """
        Save metrics to GCS.

        Args:
            metrics: Metrics dictionary
            filename: Name of metrics file

        Returns:
            True if save succeeded
        """
        metrics_path = self.local_cache_dir / filename

        # Add timestamp
        full_metrics = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
            **metrics
        }

        # Save locally
        with open(metrics_path, 'w') as f:
            json.dump(full_metrics, f, indent=2)

        # Upload to GCS
        return self.upload_file(str(metrics_path), f"metrics/{filename}", async_upload=False)

    def load_config(self, experiment_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load configuration from GCS.

        Args:
            experiment_id: Experiment ID to load (uses current if None)

        Returns:
            Configuration dictionary or None if not found
        """
        if experiment_id:
            old_exp_id = self.experiment_id
            self.experiment_id = experiment_id

        config_path = self.local_cache_dir / f"config_{experiment_id or self.experiment_id}.json"

        if self.download_file("config.json", str(config_path)):
            with open(config_path, 'r') as f:
                config = json.load(f)

            if experiment_id:
                self.experiment_id = old_exp_id

            return config

        if experiment_id:
            self.experiment_id = old_exp_id

        return None

    def list_experiments(self, prefix: str = "experiments/") -> List[str]:
        """
        List all experiments in the bucket.

        Returns:
            List of experiment IDs
        """
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix, delimiter='/')

        # Get unique experiment directories
        experiments = set()
        for blob in blobs:
            parts = blob.name.split('/')
            if len(parts) >= 2 and parts[0] == "experiments":
                experiments.add(parts[1])

        return sorted(list(experiments))

    def load_experiment_metrics(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Load metrics for a specific experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Metrics dictionary or None if not found
        """
        old_exp_id = self.experiment_id
        self.experiment_id = experiment_id

        metrics_path = self.local_cache_dir / f"metrics_{experiment_id}.json"

        success = self.download_file("metrics/final_metrics.json", str(metrics_path))

        self.experiment_id = old_exp_id

        if success and metrics_path.exists():
            with open(metrics_path, 'r') as f:
                return json.load(f)

        return None

    def get_best_checkpoint_path(self, experiment_id: Optional[str] = None) -> Optional[str]:
        """
        Get the path to the best checkpoint for an experiment.

        Args:
            experiment_id: Experiment ID (uses current if None)

        Returns:
            GCS path to best checkpoint or None
        """
        exp_id = experiment_id or self.experiment_id
        checkpoint_path = f"experiments/{exp_id}/checkpoints/best_model_improved_full.pt"

        blob = self.bucket.blob(checkpoint_path)
        if blob.exists():
            return checkpoint_path

        # Try alternative name
        checkpoint_path = f"experiments/{exp_id}/checkpoints/best_model_improved.pt"
        blob = self.bucket.blob(checkpoint_path)
        if blob.exists():
            return checkpoint_path

        return None

    def download_best_checkpoint(
        self,
        experiment_id: str,
        local_path: str = "downloaded_checkpoint.pt"
    ) -> bool:
        """
        Download the best checkpoint from an experiment.

        Args:
            experiment_id: Experiment ID
            local_path: Local destination path

        Returns:
            True if download succeeded
        """
        checkpoint_path = self.get_best_checkpoint_path(experiment_id)

        if not checkpoint_path:
            logger.error(f"No checkpoint found for experiment {experiment_id}")
            return False

        try:
            blob = self.bucket.blob(checkpoint_path)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded checkpoint to {local_path}")
            return True
        except GoogleCloudError as e:
            logger.error(f"Failed to download checkpoint: {e}")
            return False

    def wait_for_uploads(self, timeout: Optional[float] = None):
        """
        Wait for all async uploads to complete.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
        """
        for thread in self._upload_threads:
            if thread.is_alive():
                thread.join(timeout=timeout)

        # Clear finished threads
        self._upload_threads = [t for t in self._upload_threads if t.is_alive()]

        if not self._upload_threads:
            logger.info("All uploads completed")

    def check_resume_possible(self) -> bool:
        """
        Check if the current experiment can be resumed from GCS.

        Returns:
            True if checkpoints exist in GCS
        """
        # Check for any checkpoint
        checkpoint_dir = f"{self.get_experiment_path()}/checkpoints/"
        blobs = list(self.client.list_blobs(self.bucket_name, prefix=checkpoint_dir))
        return len(blobs) > 0

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the latest checkpoint file name for current experiment.

        Returns:
            Checkpoint filename (e.g., "checkpoint_750000.pt") or None
        """
        checkpoint_dir = f"{self.get_experiment_path()}/checkpoints/"
        blobs = self.client.list_blobs(self.bucket_name, prefix=checkpoint_dir)

        checkpoint_files = []
        for blob in blobs:
            filename = blob.name.split('/')[-1]
            if filename.startswith('checkpoint_') and filename.endswith('.pt'):
                # Extract step number
                try:
                    steps = int(filename.replace('checkpoint_', '').replace('.pt', ''))
                    checkpoint_files.append((steps, filename))
                except ValueError:
                    continue

        if checkpoint_files:
            # Return the checkpoint with most steps
            checkpoint_files.sort(reverse=True)
            return checkpoint_files[0][1]

        return None

    def __repr__(self) -> str:
        return f"GCSManager(bucket={self.bucket_name}, experiment={self.experiment_id})"
