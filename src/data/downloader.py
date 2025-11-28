"""
Data download module for Amazon ML Price Prediction.

This module provides the ResumableImageDownloader class for downloading
product images with resume capability, parallel processing, retry logic,
and image verification.
"""

import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image
import pandas as pd
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResumableImageDownloader:
    """
    Manages batch downloads of product images with resume capability.
    
    Features:
    - Parallel downloads with configurable workers
    - Automatic retry with exponential backoff
    - Progress tracking and resume capability
    - Image verification
    """
    
    def __init__(
        self,
        image_dir: Path,
        progress_file: Optional[Path] = None,
        timeout: int = 10,
        max_retries: int = 3
    ):
        """
        Initialize the downloader.
        
        Args:
            image_dir: Directory to save downloaded images
            progress_file: Path to progress tracking JSON file
            timeout: Timeout for each download request in seconds
            max_retries: Maximum number of retry attempts per image
        """
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
        self.progress_file = progress_file or (self.image_dir.parent / "download_progress.json")
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Load existing progress
        self.progress = self._load_progress()
    
    def download_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 5000,
        max_workers: int = 40
    ) -> Dict:
        """
        Download images in batches with parallel workers.
        
        Args:
            df: DataFrame with 'sample_id' and 'image_link' columns
            batch_size: Number of images per batch
            max_workers: Number of parallel download workers
        
        Returns:
            Dictionary with download statistics:
            - success: Number of successfully downloaded images
            - failed: Number of failed downloads
            - skipped: Number of already downloaded images
        """
        stats = {
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # Filter out already downloaded images
        to_download = []
        for _, row in df.iterrows():
            sample_id = str(row['sample_id'])
            image_link = str(row['image_link'])
            
            # Check if already downloaded
            if sample_id in self.progress.get('completed', set()):
                stats['skipped'] += 1
                continue
            
            # Check if file exists
            image_path = self.image_dir / f"{sample_id}.jpg"
            if image_path.exists():
                self.progress.setdefault('completed', set()).add(sample_id)
                stats['skipped'] += 1
                continue
            
            to_download.append((sample_id, image_link))
        
        if not to_download:
            logger.info("No images to download")
            return stats
        
        logger.info(f"Downloading {len(to_download)} images with {max_workers} workers")
        
        # Process in batches
        for batch_start in range(0, len(to_download), batch_size):
            batch_end = min(batch_start + batch_size, len(to_download))
            batch = to_download[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: "
                       f"images {batch_start} to {batch_end}")
            
            # Download batch in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._download_single,
                        sample_id,
                        url,
                        self.timeout,
                        self.max_retries
                    ): (sample_id, url)
                    for sample_id, url in batch
                }
                
                # Process completed downloads with progress bar
                with tqdm(total=len(batch), desc=f"Batch {batch_start//batch_size + 1}") as pbar:
                    for future in as_completed(futures):
                        sample_id, url = futures[future]
                        try:
                            result_id, success = future.result()
                            
                            if success:
                                stats['success'] += 1
                                self.progress.setdefault('completed', set()).add(sample_id)
                            else:
                                stats['failed'] += 1
                                self.progress.setdefault('failed', set()).add(sample_id)
                        
                        except Exception as e:
                            logger.error(f"Error downloading {sample_id}: {e}")
                            stats['failed'] += 1
                            self.progress.setdefault('failed', set()).add(sample_id)
                        
                        pbar.update(1)
            
            # Save progress after each batch
            self._save_progress()
            logger.info(f"Batch complete. Success: {stats['success']}, "
                       f"Failed: {stats['failed']}, Skipped: {stats['skipped']}")
        
        logger.info(f"Download complete. Total - Success: {stats['success']}, "
                   f"Failed: {stats['failed']}, Skipped: {stats['skipped']}")
        
        return stats
    
    def _download_single(
        self,
        sample_id: str,
        url: str,
        timeout: int = 10,
        max_retries: int = 3
    ) -> Tuple[str, bool]:
        """
        Download a single image with retry logic and exponential backoff.
        
        Args:
            sample_id: Unique identifier for the image
            url: URL to download from
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        
        Returns:
            Tuple of (sample_id, success_flag)
        """
        image_path = self.image_dir / f"{sample_id}.jpg"
        
        # Skip if already exists
        if image_path.exists():
            return sample_id, True
        
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Download image
                response = requests.get(url, timeout=timeout, stream=True)
                response.raise_for_status()
                
                # Save image
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify image can be opened
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                    return sample_id, True
                
                except Exception as e:
                    # Image is corrupted, delete it
                    if image_path.exists():
                        image_path.unlink()
                    raise ValueError(f"Corrupted image: {e}")
            
            except Exception as e:
                last_error = e
                retry_count += 1
                
                if retry_count < max_retries:
                    # Exponential backoff: 2^retry_count seconds
                    wait_time = 2 ** retry_count
                    time.sleep(wait_time)
                    logger.debug(f"Retry {retry_count}/{max_retries} for {sample_id} "
                               f"after {wait_time}s delay")
        
        # All retries failed
        logger.warning(f"Failed to download {sample_id} after {max_retries} attempts: "
                      f"{last_error}")
        return sample_id, False
    
    def _save_progress(self) -> None:
        """Save download progress to JSON file."""
        # Convert sets to lists for JSON serialization
        progress_data = {
            'completed': list(self.progress.get('completed', set())),
            'failed': list(self.progress.get('failed', set()))
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f)
    
    def _load_progress(self) -> Dict:
        """
        Load previous download progress.
        
        Returns:
            Dictionary with 'completed' and 'failed' sets
        """
        if not self.progress_file.exists():
            return {'completed': set(), 'failed': set()}
        
        try:
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
            
            # Convert lists back to sets
            return {
                'completed': set(progress_data.get('completed', [])),
                'failed': set(progress_data.get('failed', []))
            }
        
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
            return {'completed': set(), 'failed': set()}
    
    def verify_images(self, sample_ids: Optional[List[str]] = None) -> Dict:
        """
        Verify image integrity for downloaded images.
        
        Args:
            sample_ids: List of sample IDs to verify. If None, verify all.
        
        Returns:
            Dictionary with verification results:
            - valid: Number of valid images
            - corrupted: List of corrupted image sample IDs
        """
        results = {
            'valid': 0,
            'corrupted': []
        }
        
        # Get list of images to verify
        if sample_ids is None:
            image_files = list(self.image_dir.glob("*.jpg"))
            sample_ids = [f.stem for f in image_files]
        
        logger.info(f"Verifying {len(sample_ids)} images")
        
        for sample_id in tqdm(sample_ids, desc="Verifying images"):
            image_path = self.image_dir / f"{sample_id}.jpg"
            
            if not image_path.exists():
                results['corrupted'].append(sample_id)
                continue
            
            try:
                with Image.open(image_path) as img:
                    img.verify()
                
                # Re-open to check if it can be loaded (verify() closes the file)
                with Image.open(image_path) as img:
                    img.load()
                
                results['valid'] += 1
            
            except Exception as e:
                logger.warning(f"Corrupted image {sample_id}: {e}")
                results['corrupted'].append(sample_id)
        
        logger.info(f"Verification complete. Valid: {results['valid']}, "
                   f"Corrupted: {len(results['corrupted'])}")
        
        return results
    
    def get_download_stats(self) -> Dict:
        """
        Get current download statistics.
        
        Returns:
            Dictionary with download statistics
        """
        return {
            'completed': len(self.progress.get('completed', set())),
            'failed': len(self.progress.get('failed', set())),
            'total_processed': len(self.progress.get('completed', set())) + 
                             len(self.progress.get('failed', set()))
        }
