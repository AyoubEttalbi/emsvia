import os
import shutil
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def organize_dataset(source_dir: str, dest_dir: str, dry_run: bool = False, move: bool = False):
    """
    Organizes images from source_dir into dest_dir by student name.
    Expects filenames like: Name.Part1.Part2.Part3.jpg
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not source_path.exists():
        logger.error(f"Source directory {source_dir} does not exist.")
        return

    # Ensure destination exists
    if not dry_run:
        dest_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning {source_path}...")
    
    # Supported image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    
    files = [f for f in source_path.iterdir() if f.is_file() and f.suffix.lower() in extensions]
    logger.info(f"Found {len(files)} images to process.")

    stats = {} # student_name -> count
    
    for file in files:
        # Extract student name (first part before the first dot)
        student_name = file.name.split('.')[0]
        
        student_dir = dest_path / student_name
        
        if not dry_run:
            student_dir.mkdir(parents=True, exist_ok=True)
            
        target_file = student_dir / file.name
        
        action = "Moving" if move else "Copying"
        if dry_run:
            logger.info(f"[DRY-RUN] Would {action.lower()} {file.name} -> {student_name}/")
        else:
            try:
                if move:
                    shutil.move(str(file), str(target_file))
                else:
                    shutil.copy2(str(file), str(target_file))
            except Exception as e:
                logger.error(f"Failed to {action.lower()} {file.name}: {e}")
                continue
        
        stats[student_name] = stats.get(student_name, 0) + 1

    # Print summary
    logger.info("=" * 40)
    logger.info("ORGANIZATION SUMMARY")
    logger.info("=" * 40)
    for name, count in sorted(stats.items()):
        logger.info(f"  {name}: {count} images")
    logger.info("-" * 40)
    logger.info(f"Total students: {len(stats)}")
    logger.info(f"Total images: {sum(stats.values())}")
    if dry_run:
        logger.info("NOTE: This was a DRY-RUN. No files were actually moved/copied.")
    logger.info("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize unstructured student dataset into folders.")
    parser.add_argument("--source", type=str, default="data/Smart attendance dataset/train", help="Source directory containing images.")
    parser.add_argument("--dest", type=str, default="data/student_images", help="Destination directory for organized folders.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes.")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them.")
    
    args = parser.parse_args()
    
    # Resolve relative paths relative to project root (parent of scripts/)
    base_dir = Path(__file__).resolve().parent.parent
    
    source = args.source
    if not os.path.isabs(source):
        source = str(base_dir / source)
        
    dest = args.dest
    if not os.path.isabs(dest):
        dest = str(base_dir / dest)

    organize_dataset(source, dest, dry_run=args.dry_run, move=args.move)
