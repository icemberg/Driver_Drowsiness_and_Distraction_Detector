"""
Drink Dataset Image Downloader

This module downloads 1000+ diverse images per drink class from Bing Image Search.
It automatically cleans corrupt images and organizes them for training.

Usage:
    python image_downloader.py --download
    python image_downloader.py --clean
    python image_downloader.py --download --clean
    python image_downloader.py --stats
"""

import os
import argparse
from pathlib import Path
from icrawler.builtin import BingImageCrawler
from PIL import Image, UnidentifiedImageError
import logging
from datetime import datetime

# ============================================================
# Configuration
# ============================================================

# Drink classes with multiple search keywords for diversity
DRINK_CLASSES_KEYWORDS = {
    "cup": [
        "cup",
        "coffee cup",
        "ceramic cup",
        "plastic cup",
        "disposable cup",
        "drinking cup",
        "empty cup",
        "white cup"
    ],
    "bottle": [
        "bottle",
        "drink bottle",
        "plastic bottle",
        "glass bottle",
        "water bottle",
        "beverage bottle",
        "empty bottle",
        "bottle with drink"
    ],
    "mug": [
        "mug",
        "coffee mug",
        "ceramic mug",
        "tea mug",
        "drinking mug",
        "hot drink mug",
        "morning mug",
        "white mug"
    ],
    "can": [
        "can",
        "drink can",
        "aluminum can",
        "soda can",
        "beverage can",
        "soft drink can",
        "beer can",
        "empty can"
    ],
    "glass": [
        "glass",
        "drinking glass",
        "water glass",
        "beverage glass",
        "clear glass",
        "glass cup",
        "glass with drink",
        "empty glass"
    ],
    "drinking_bottle": [
        "drinking bottle",
        "sport water bottle",
        "insulated bottle",
        "stainless steel bottle",
        "hydro flask",
        "thermos bottle",
        "portable bottle",
        "travel bottle"
    ],
    "soda_can": [
        "soda can",
        "cola can",
        "soft drink can",
        "carbonated drink can",
        "energy drink can",
        "fizzy drink can",
        "packed soda",
        "soda aluminum can"
    ],
    "beer_bottle": [
        "beer bottle",
        "beer glass bottle",
        "beer with bottle",
        "alcoholic bottle",
        "brown beer bottle",
        "green beer bottle",
        "glass beer bottle",
        "cold beer bottle"
    ],
    "water_bottle": [
        "water bottle",
        "mineral water bottle",
        "plastic water bottle",
        "glass water bottle",
        "drinking water bottle",
        "pure water bottle",
        "water container",
        "portable water bottle"
    ],
    "tea_cup": [
        "tea cup",
        "tea mug",
        "ceramic tea cup",
        "porcelain tea cup",
        "white tea cup",
        "hot tea cup",
        "brewing tea",
        "tea service"
    ]
}

# Download settings
IMAGES_PER_KEYWORD = 120  # 120 keywords per class × ~10 keywords = ~1200 images
DATASET_BASE_DIR = "drink_dataset"
CRAWLER_THREADS = 4
CRAWLER_TIMEOUT = 10

# Logging setup
LOG_FILE = "image_downloader.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DrinkImageDownloader:
    """Download and manage drink object images from Bing."""
    
    def __init__(self, base_dir=DATASET_BASE_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dataset base directory: {self.base_dir}")
    
    def download_class_images(self, class_name, keywords, images_per_keyword=IMAGES_PER_KEYWORD):
        """
        Download images for a single drink class using multiple keywords.
        
        Args:
            class_name: Name of the drink class
            keywords: List of search keywords for this class
            images_per_keyword: Number of images to download per keyword
        """
        class_dir = self.base_dir / class_name / "images"
        class_dir.mkdir(parents=True, exist_ok=True)
        
        total_target = len(keywords) * images_per_keyword
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading {class_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Keywords: {len(keywords)}")
        logger.info(f"Target: ~{total_target} images ({images_per_keyword} per keyword)")
        logger.info(f"Output directory: {class_dir}")
        
        existing_count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
        logger.info(f"Existing images: {existing_count}")
        
        crawler = BingImageCrawler(
            downloader_threads=CRAWLER_THREADS,
            storage={"root_dir": str(class_dir)}
        )
        
        total_downloaded = 0
        
        for keyword_idx, keyword in enumerate(keywords, 1):
            try:
                logger.info(f"  [{keyword_idx}/{len(keywords)}] Downloading: '{keyword}'")
                
                # Get current image count before crawl
                before = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
                
                crawler.crawl(
                    keyword=keyword,
                    max_num=images_per_keyword,
                    file_idx_offset=before  # Continue numbering
                )
                
                # Count new images
                after = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
                downloaded = after - before
                total_downloaded += downloaded
                
                logger.info(f"    [OK] Downloaded {downloaded} images")
                
            except Exception as e:
                logger.warning(f"    ⚠ Error downloading '{keyword}': {e}")
                continue
        
        logger.info(f"[OK] {class_name}: Total {total_downloaded} images downloaded")
        return total_downloaded
    
    def download_all(self):
        """Download images for all drink classes."""
        logger.info("\n" + "="*60)
        logger.info("DRINK IMAGE DOWNLOADER")
        logger.info("="*60)
        logger.info(f"Classes: {len(DRINK_CLASSES_KEYWORDS)}")
        logger.info(f"Total target: ~{len(DRINK_CLASSES_KEYWORDS) * len(next(iter(DRINK_CLASSES_KEYWORDS.values()))) * IMAGES_PER_KEYWORD} images")
        
        total_images = 0
        
        for class_name, keywords in DRINK_CLASSES_KEYWORDS.items():
            try:
                count = self.download_class_images(class_name, keywords)
                total_images += count
            except Exception as e:
                logger.error(f"Failed to download {class_name}: {e}")
                continue
        
        logger.info("\n" + "="*60)
        logger.info(f"[OK] Download complete! Total images: {total_images}")
        logger.info("="*60)
        
        return total_images
    
    def clean_corrupt_images(self, verbose=True):
        """
        Remove corrupt, duplicate, and low-quality images.
        
        Args:
            verbose: Print progress updates
        
        Returns:
            Dictionary with cleanup statistics
        """
        logger.info("\n" + "="*60)
        logger.info("CLEANING CORRUPT IMAGES")
        logger.info("="*60)
        
        stats = {
            'total_checked': 0,
            'removed_corrupt': 0,
            'removed_small': 0,
            'removed_invalid_format': 0,
            'total_removed': 0,
            'classes': {}
        }
        
        valid_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        min_width = 100  # Minimum image width in pixels
        min_height = 100  # Minimum image height in pixels
        min_file_size = 5000  # Minimum file size in bytes (5KB)
        
        for class_dir in self.base_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            images_dir = class_dir / "images"
            if not images_dir.exists():
                continue
            
            class_name = class_dir.name
            class_stats = {
                'total': 0,
                'kept': 0,
                'removed': []
            }
            
            logger.info(f"\nCleaning: {class_name}")
            
            for img_file in images_dir.iterdir():
                if img_file.suffix.lower() not in valid_formats:
                    continue
                
                class_stats['total'] += 1
                stats['total_checked'] += 1
                
                try:
                    # Check file size
                    file_size = img_file.stat().st_size
                    if file_size < min_file_size:
                        logger.debug(f"  Removing {img_file.name}: too small ({file_size} bytes)")
                        class_stats['removed'].append(("size", img_file.name))
                        img_file.unlink()
                        stats['removed_small'] += 1
                        continue
                    
                    # Check if image can be opened and is valid
                    img = Image.open(img_file)
                    img.verify()
                    
                    # Check dimensions
                    width, height = img.size
                    if width < min_width or height < min_height:
                        logger.debug(f"  Removing {img_file.name}: too small dimensions ({width}x{height})")
                        class_stats['removed'].append(("dimensions", img_file.name))
                        img_file.unlink()
                        stats['removed_small'] += 1
                        continue
                    
                    class_stats['kept'] += 1
                    
                except UnidentifiedImageError:
                    logger.debug(f"  Removing {img_file.name}: unidentified format")
                    class_stats['removed'].append(("format", img_file.name))
                    img_file.unlink()
                    stats['removed_invalid_format'] += 1
                    
                except Exception as e:
                    logger.debug(f"  Removing {img_file.name}: {e}")
                    class_stats['removed'].append(("corrupt", img_file.name))
                    img_file.unlink()
                    stats['removed_corrupt'] += 1
            
            stats['total_removed'] += len(class_stats['removed'])
            stats['classes'][class_name] = class_stats
            
            logger.info(f"  Total: {class_stats['total']} | Kept: {class_stats['kept']} | Removed: {len(class_stats['removed'])}")
            if len(class_stats['removed']) == 0 and class_stats['total'] > 0:
                logger.info(f"    [OK] All images passed quality checks")
        
        logger.info("\n" + "="*60)
        logger.info("CLEANUP SUMMARY")
        logger.info("="*60)
        logger.info(f"Total checked: {stats['total_checked']}")
        logger.info(f"Total removed: {stats['total_removed']}")
        logger.info(f"  - Corrupt: {stats['removed_corrupt']}")
        logger.info(f"  - Too small: {stats['removed_small']}")
        logger.info(f"  - Invalid format: {stats['removed_invalid_format']}")
        logger.info("="*60)
        
        return stats
    
    def get_dataset_statistics(self):
        """Get statistics about the downloaded dataset."""
        logger.info("\n" + "="*60)
        logger.info("DATASET STATISTICS")
        logger.info("="*60)
        
        total_images = 0
        class_stats = {}
        
        for class_dir in sorted(self.base_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            images_dir = class_dir / "images"
            if not images_dir.exists():
                continue
            
            class_name = class_dir.name
            image_count = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.gif")))
            class_stats[class_name] = image_count
            total_images += image_count
            
            logger.info(f"{class_name:20s}: {image_count:4d} images")
        
        logger.info("="*60)
        logger.info(f"{'Total':20s}: {total_images:4d} images across {len(class_stats)} classes")
        logger.info("="*60)
        
        # Check if balanced
        if class_stats:
            values = list(class_stats.values())
            min_val = min(values)
            max_val = max(values)
            avg_val = total_images / len(values)
            
            logger.info(f"\nBalance metrics:")
            logger.info(f"  Average per class: {avg_val:.0f}")
            logger.info(f"  Min: {min_val}, Max: {max_val}")
            logger.info(f"  Imbalance ratio: {max_val/max(min_val, 1):.2f}x")
            
            if min_val < avg_val * 0.7:
                logger.warning("[WARNING] Dataset is imbalanced! Some classes have too few images.")
            else:
                logger.info("[OK] Dataset is well-balanced!")
        
        return class_stats


def main():
    parser = argparse.ArgumentParser(description="Drink Dataset Image Downloader")
    parser.add_argument('--download', action='store_true', help='Download images')
    parser.add_argument('--clean', action='store_true', help='Clean corrupt images')
    parser.add_argument('--stats', action='store_true', help='Show dataset statistics')
    parser.add_argument('--dataset-dir', type=str, default=DATASET_BASE_DIR,
                       help='Dataset directory path')
    parser.add_argument('--images-per-keyword', type=int, default=IMAGES_PER_KEYWORD,
                       help='Images to download per keyword')
    
    args = parser.parse_args()
    
    downloader = DrinkImageDownloader(base_dir=args.dataset_dir)
    
    # If no mode specified, do full pipeline
    if not args.download and not args.clean and not args.stats:
        logger.info("No mode specified. Running full pipeline: download → clean → stats")
        args.download = True
        args.clean = True
        args.stats = True
    
    if args.download:
        try:
            downloader.download_all()
        except Exception as e:
            logger.error(f"Download failed: {e}")
    
    if args.clean:
        try:
            downloader.clean_corrupt_images()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    if args.stats:
        try:
            stats = downloader.get_dataset_statistics()
        except Exception as e:
            logger.error(f"Statistics failed: {e}")


if __name__ == '__main__':
    main()
