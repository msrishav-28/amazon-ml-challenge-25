"""
Property-based tests for data download module.

These tests verify batch download consistency, resume from interruption,
retry with exponential backoff, and corrupted image detection as specified
in the design document.
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume
import pandas as pd
import sys
import os
from io import BytesIO
from PIL import Image
import requests

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.downloader import ResumableImageDownloader


# Helper strategies for generating test data
@st.composite
def sample_dataframe_strategy(draw, min_size=1, max_size=20):
    """Generate a random DataFrame with sample_id and image_link columns."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    
    sample_ids = [f"sample_{i}" for i in range(size)]
    image_links = [f"http://example.com/image_{i}.jpg" for i in range(size)]
    
    return pd.DataFrame({
        'sample_id': sample_ids,
        'image_link': image_links
    })


@st.composite
def batch_size_strategy(draw, df_size):
    """Generate a valid batch size for a given DataFrame size."""
    # Batch size should be at least 1 and at most df_size
    if df_size == 0:
        return 1
    return draw(st.integers(min_value=1, max_value=max(1, df_size * 2)))


def create_mock_image_response(success=True, corrupted=False):
    """Create a mock response for image download."""
    response = Mock()
    response.status_code = 200 if success else 404
    
    if success:
        if corrupted:
            # Create invalid image data
            response.iter_content = Mock(return_value=[b'corrupted_data'])
        else:
            # Create a valid small image
            img = Image.new('RGB', (10, 10), color='red')
            img_bytes = BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            response.iter_content = Mock(return_value=[img_bytes.read()])
    else:
        response.raise_for_status = Mock(side_effect=requests.HTTPError("404"))
    
    return response


class TestDownloaderProperties:
    """Property-based tests for data download module."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.temp_dir) / "images"
        self.progress_file = Path(self.temp_dir) / "progress.json"
    
    def teardown_method(self):
        """Clean up temporary directory after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @given(
        df=sample_dataframe_strategy(min_size=5, max_size=15)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_1_batch_download_consistency(self, df):
        """
        Feature: amazon-ml-price-prediction, Property 1: Batch download consistency
        
        For any DataFrame with image URLs and any batch size, downloading in batches
        should result in the same set of successfully downloaded images as downloading
        all at once (order-independent).
        
        Validates: Requirements 1.1
        """
        df_size = len(df)
        assume(df_size > 0)
        
        # Create fresh temp directories for this example
        temp_dir1 = tempfile.mkdtemp()
        temp_dir2 = tempfile.mkdtemp()
        
        try:
            # Create two downloaders with different batch sizes
            downloader1 = ResumableImageDownloader(
                image_dir=Path(temp_dir1) / "images",
                progress_file=Path(temp_dir1) / "progress.json",
                timeout=10,
                max_retries=3
            )
            
            downloader2 = ResumableImageDownloader(
                image_dir=Path(temp_dir2) / "images",
                progress_file=Path(temp_dir2) / "progress.json",
                timeout=10,
                max_retries=3
            )
            
            # Mock the download to avoid actual network calls
            # We need to mock both requests.get and PIL.Image.open
            with patch('src.data.downloader.requests.get') as mock_get, \
                 patch('src.data.downloader.Image.open') as mock_image_open:
                
                mock_get.return_value = create_mock_image_response(success=True)
                
                # Mock PIL.Image.open to return a valid image mock
                mock_img = MagicMock()
                mock_img.verify = MagicMock()
                mock_img.load = MagicMock()
                mock_img.__enter__ = MagicMock(return_value=mock_img)
                mock_img.__exit__ = MagicMock(return_value=False)
                mock_image_open.return_value = mock_img
                
                # Download with batch size = 3
                stats1 = downloader1.download_batch(df, batch_size=3, max_workers=2)
                
                # Download with batch size = 7 (different batch size)
                stats2 = downloader2.download_batch(df, batch_size=7, max_workers=2)
            
            # Both should have downloaded the same number of images
            assert stats1['success'] == stats2['success'], \
                f"Different batch sizes produced different success counts: " \
                f"{stats1['success']} vs {stats2['success']}"
            
            # Both should have the same set of downloaded files
            files1 = set(f.name for f in downloader1.image_dir.glob("*.jpg"))
            files2 = set(f.name for f in downloader2.image_dir.glob("*.jpg"))
            
            assert files1 == files2, \
                f"Different batch sizes produced different file sets: " \
                f"{files1} vs {files2}"
            
            # Total processed should equal DataFrame size
            assert stats1['success'] + stats1['failed'] == df_size
            assert stats2['success'] + stats2['failed'] == df_size
        
        finally:
            # Clean up temp directories
            shutil.rmtree(temp_dir1, ignore_errors=True)
            shutil.rmtree(temp_dir2, ignore_errors=True)
    
    @given(
        df=sample_dataframe_strategy(min_size=10, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_2_resume_from_interruption(self, df):
        """
        Feature: amazon-ml-price-prediction, Property 2: Resume from interruption
        
        For any download progress state, resuming a download should skip
        already-downloaded images and only download remaining images,
        resulting in the complete set without duplicates.
        
        Validates: Requirements 1.2
        """
        df_size = len(df)
        assume(df_size >= 10)
        
        # Create fresh temp directory for this example
        temp_dir = tempfile.mkdtemp()
        
        try:
            downloader = ResumableImageDownloader(
                image_dir=Path(temp_dir) / "images",
                progress_file=Path(temp_dir) / "progress.json",
                timeout=10,
                max_retries=3
            )
            
            # Split DataFrame into two parts to simulate interruption
            split_point = df_size // 2
            df_part1 = df.iloc[:split_point].copy()
            df_part2 = df.iloc[split_point:].copy()
            
            with patch('src.data.downloader.requests.get') as mock_get, \
                 patch('src.data.downloader.Image.open') as mock_image_open:
                
                mock_get.return_value = create_mock_image_response(success=True)
                
                # Mock PIL.Image.open to return a valid image mock
                mock_img = MagicMock()
                mock_img.verify = MagicMock()
                mock_img.load = MagicMock()
                mock_img.__enter__ = MagicMock(return_value=mock_img)
                mock_img.__exit__ = MagicMock(return_value=False)
                mock_image_open.return_value = mock_img
                
                # Download first part
                stats1 = downloader.download_batch(df_part1, batch_size=5, max_workers=2)
                
                # Get files after first download
                files_after_part1 = set(f.name for f in downloader.image_dir.glob("*.jpg"))
                
                # Resume with full DataFrame (simulating resume after interruption)
                stats2 = downloader.download_batch(df, batch_size=5, max_workers=2)
            
            # Second download should skip already downloaded images
            assert stats2['skipped'] >= stats1['success'], \
                f"Resume didn't skip already downloaded images: " \
                f"skipped={stats2['skipped']}, previous_success={stats1['success']}"
            
            # Total unique files should equal DataFrame size
            final_files = set(f.name for f in downloader.image_dir.glob("*.jpg"))
            assert len(final_files) == df_size, \
                f"Resume produced wrong number of files: {len(final_files)} vs {df_size}"
            
            # No duplicates should be created
            expected_files = {f"{row['sample_id']}.jpg" for _, row in df.iterrows()}
            assert final_files == expected_files, \
                "Resume created unexpected files or missed some"
            
            # Files from first download should still exist
            assert files_after_part1.issubset(final_files), \
                "Resume deleted previously downloaded files"
        
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @given(
        sample_id=st.text(min_size=1, max_size=20, 
                         alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        max_retries=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_3_retry_with_exponential_backoff(self, sample_id, max_retries):
        """
        Feature: amazon-ml-price-prediction, Property 3: Retry with exponential backoff
        
        For any failed image download, the system should retry exactly max_retries times
        with increasing delays, and the delay between retry N and N+1 should be greater
        than the delay between N-1 and N.
        
        Validates: Requirements 1.3
        """
        downloader = ResumableImageDownloader(
            image_dir=self.image_dir,
            progress_file=self.progress_file,
            timeout=10,
            max_retries=max_retries
        )
        
        # Track retry attempts and delays
        retry_times = []
        
        def mock_get_with_failure(*args, **kwargs):
            """Mock that always fails to trigger retries."""
            retry_times.append(time.time())
            raise requests.Timeout("Simulated timeout")
        
        with patch('src.data.downloader.requests.get', side_effect=mock_get_with_failure):
            with patch('src.data.downloader.time.sleep') as mock_sleep:
                # Attempt download (will fail and retry)
                result_id, success = downloader._download_single(
                    sample_id=sample_id,
                    url="http://example.com/test.jpg",
                    timeout=10,
                    max_retries=max_retries
                )
        
        # Should have failed
        assert not success, "Download should have failed after all retries"
        
        # Should have attempted exactly max_retries times (initial + retries)
        assert len(retry_times) == max_retries, \
            f"Expected {max_retries} attempts, got {len(retry_times)}"
        
        # Verify exponential backoff was applied
        # mock_sleep should be called (max_retries - 1) times
        assert mock_sleep.call_count == max_retries - 1, \
            f"Expected {max_retries - 1} sleep calls, got {mock_sleep.call_count}"
        
        # Verify delays are exponentially increasing
        sleep_delays = [call[0][0] for call in mock_sleep.call_args_list]
        
        for i in range(len(sleep_delays) - 1):
            current_delay = sleep_delays[i]
            next_delay = sleep_delays[i + 1]
            
            # Each delay should be greater than the previous (exponential backoff)
            assert next_delay > current_delay, \
                f"Delay not increasing exponentially: {current_delay} -> {next_delay}"
            
            # Verify it follows 2^retry_count pattern
            expected_delay = 2 ** (i + 1)
            assert current_delay == expected_delay, \
                f"Delay doesn't match exponential pattern: " \
                f"expected {expected_delay}, got {current_delay}"
    
    @given(
        df=sample_dataframe_strategy(min_size=5, max_size=15)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_4_corrupted_image_detection(self, df):
        """
        Feature: amazon-ml-price-prediction, Property 4: Corrupted image detection
        
        For any set of downloaded images, running verification should correctly
        identify all images that cannot be opened by PIL.Image and report them
        as corrupted.
        
        Validates: Requirements 1.4
        """
        df_size = len(df)
        assume(df_size >= 5)
        
        downloader = ResumableImageDownloader(
            image_dir=self.image_dir,
            progress_file=self.progress_file,
            timeout=10,
            max_retries=3
        )
        
        # Determine which images will be corrupted (every 3rd image)
        corrupted_indices = set(range(0, df_size, 3))
        expected_corrupted = {f"sample_{i}" for i in corrupted_indices}
        
        def mock_get_variable(*args, **kwargs):
            """Mock that returns corrupted images for specific indices."""
            url = args[0]
            # Extract index from URL
            idx = int(url.split('_')[-1].split('.')[0])
            
            if idx in corrupted_indices:
                return create_mock_image_response(success=True, corrupted=True)
            else:
                return create_mock_image_response(success=True, corrupted=False)
        
        with patch('src.data.downloader.requests.get', side_effect=mock_get_variable):
            # Download images (some will be corrupted)
            stats = downloader.download_batch(df, batch_size=5, max_workers=2)
        
        # Verify images
        verification_results = downloader.verify_images()
        
        # All corrupted images should be detected
        detected_corrupted = set(verification_results['corrupted'])
        
        # The corrupted images should have failed during download
        # (because we verify after download in _download_single)
        # So they shouldn't exist as files
        
        # For this test, let's manually create some corrupted files
        # to test the verification function
        for idx in corrupted_indices:
            sample_id = f"sample_{idx}"
            image_path = downloader.image_dir / f"{sample_id}.jpg"
            # Create a corrupted file
            with open(image_path, 'wb') as f:
                f.write(b'corrupted_data')
        
        # Now verify again
        verification_results = downloader.verify_images()
        detected_corrupted = set(verification_results['corrupted'])
        
        # All corrupted images should be detected
        assert expected_corrupted.issubset(detected_corrupted), \
            f"Not all corrupted images detected. " \
            f"Expected: {expected_corrupted}, Detected: {detected_corrupted}"
        
        # Valid images should not be marked as corrupted
        valid_sample_ids = {f"sample_{i}" for i in range(df_size) 
                           if i not in corrupted_indices}
        
        for sample_id in valid_sample_ids:
            assert sample_id not in detected_corrupted, \
                f"Valid image {sample_id} incorrectly marked as corrupted"


class TestDownloaderEdgeCases:
    """Additional edge case tests for the downloader."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.temp_dir) / "images"
        self.progress_file = Path(self.temp_dir) / "progress.json"
    
    def teardown_method(self):
        """Clean up temporary directory after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_empty_dataframe_handling(self):
        """Test that empty DataFrame is handled gracefully."""
        downloader = ResumableImageDownloader(
            image_dir=self.image_dir,
            progress_file=self.progress_file
        )
        
        df = pd.DataFrame({'sample_id': [], 'image_link': []})
        
        stats = downloader.download_batch(df, batch_size=5, max_workers=2)
        
        assert stats['success'] == 0
        assert stats['failed'] == 0
        assert stats['skipped'] == 0
    
    def test_progress_persistence(self):
        """Test that progress is correctly saved and loaded."""
        downloader = ResumableImageDownloader(
            image_dir=self.image_dir,
            progress_file=self.progress_file
        )
        
        # Manually add some progress
        downloader.progress['completed'].add('sample_1')
        downloader.progress['completed'].add('sample_2')
        downloader.progress['failed'].add('sample_3')
        downloader._save_progress()
        
        # Create new downloader instance (should load progress)
        downloader2 = ResumableImageDownloader(
            image_dir=self.image_dir,
            progress_file=self.progress_file
        )
        
        assert 'sample_1' in downloader2.progress['completed']
        assert 'sample_2' in downloader2.progress['completed']
        assert 'sample_3' in downloader2.progress['failed']
    
    def test_verify_nonexistent_images(self):
        """Test verification of images that don't exist."""
        downloader = ResumableImageDownloader(
            image_dir=self.image_dir,
            progress_file=self.progress_file
        )
        
        # Verify images that don't exist
        results = downloader.verify_images(['nonexistent_1', 'nonexistent_2'])
        
        assert results['valid'] == 0
        assert len(results['corrupted']) == 2
        assert 'nonexistent_1' in results['corrupted']
        assert 'nonexistent_2' in results['corrupted']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
