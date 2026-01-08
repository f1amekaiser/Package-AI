#!/usr/bin/env python3
"""
Sample Dataset Generator

Generates synthetic sample images for testing and demo purposes.
Creates damaged and clean package images with annotations.
"""

import os
import random
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime

import numpy as np
import cv2


class SampleDatasetGenerator:
    """
    Generates synthetic package images for testing.
    
    Creates:
    - Clean package images
    - Damaged package images with various damage types
    - YOLO format annotations
    """
    
    CLASS_NAMES = [
        "structural_deformation",
        "surface_breach", 
        "contamination_stain",
        "compression_damage",
        "tape_seal_damage"
    ]
    
    # Package colors
    PACKAGE_COLORS = [
        (139, 119, 101),  # Brown cardboard
        (200, 200, 200),  # White box
        (180, 160, 140),  # Kraft paper
        (220, 200, 180),  # Light cardboard
    ]
    
    def __init__(
        self,
        output_dir: str = "data",
        image_size: Tuple[int, int] = (640, 640),
        num_clean: int = 100,
        num_damaged: int = 100
    ):
        """
        Initialize the generator.
        
        Args:
            output_dir: Output directory for generated data
            image_size: Image dimensions (width, height)
            num_clean: Number of clean images to generate
            num_damaged: Number of damaged images to generate
        """
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.num_clean = num_clean
        self.num_damaged = num_damaged
        
        # Create directories
        (self.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    def generate(self, split_ratio: float = 0.8):
        """
        Generate the complete sample dataset.
        
        Args:
            split_ratio: Ratio of training to validation data
        """
        print(f"Generating sample dataset...")
        print(f"  Clean images: {self.num_clean}")
        print(f"  Damaged images: {self.num_damaged}")
        
        all_samples = []
        
        # Generate clean images
        for i in range(self.num_clean):
            sample = self._generate_clean_image(f"clean_{i:04d}")
            all_samples.append(sample)
        
        # Generate damaged images
        for i in range(self.num_damaged):
            sample = self._generate_damaged_image(f"damaged_{i:04d}")
            all_samples.append(sample)
        
        # Shuffle and split
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * split_ratio)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
        
        # Save samples
        for sample in train_samples:
            self._save_sample(sample, "train")
        
        for sample in val_samples:
            self._save_sample(sample, "val")
        
        print(f"Generated {len(train_samples)} training, {len(val_samples)} validation samples")
        print(f"Saved to: {self.output_dir}")
        
        return {
            "train": len(train_samples),
            "val": len(val_samples),
            "total": len(all_samples)
        }
    
    def _generate_clean_image(self, name: str) -> Dict[str, Any]:
        """Generate a clean package image."""
        image = self._create_base_image()
        
        return {
            "name": name,
            "image": image,
            "annotations": []
        }
    
    def _generate_damaged_image(self, name: str) -> Dict[str, Any]:
        """Generate a damaged package image with annotations."""
        image = self._create_base_image()
        annotations = []
        
        # Add 1-3 damage instances
        num_damages = random.randint(1, 3)
        
        for _ in range(num_damages):
            damage_type = random.randint(0, 4)
            bbox = self._add_damage(image, damage_type)
            
            if bbox:
                annotations.append({
                    "class_id": damage_type,
                    "bbox": bbox
                })
        
        return {
            "name": name,
            "image": image,
            "annotations": annotations
        }
    
    def _create_base_image(self) -> np.ndarray:
        """Create a base package image."""
        w, h = self.image_size
        
        # Background (warehouse floor/conveyor)
        bg_color = random.choice([
            (60, 60, 60),    # Dark gray
            (80, 80, 80),    # Medium gray
            (100, 90, 80),   # Brown-ish
        ])
        image = np.full((h, w, 3), bg_color, dtype=np.uint8)
        
        # Package dimensions (centered, variable size)
        pkg_w = random.randint(int(w * 0.5), int(w * 0.8))
        pkg_h = random.randint(int(h * 0.5), int(h * 0.8))
        pkg_x = (w - pkg_w) // 2
        pkg_y = (h - pkg_h) // 2
        
        # Package color
        pkg_color = random.choice(self.PACKAGE_COLORS)
        
        # Draw package
        cv2.rectangle(image, (pkg_x, pkg_y), (pkg_x + pkg_w, pkg_y + pkg_h), pkg_color, -1)
        
        # Add some texture (noise)
        noise = np.random.randint(-15, 15, (pkg_h, pkg_w, 3), dtype=np.int16)
        image[pkg_y:pkg_y+pkg_h, pkg_x:pkg_x+pkg_w] = np.clip(
            image[pkg_y:pkg_y+pkg_h, pkg_x:pkg_x+pkg_w].astype(np.int16) + noise,
            0, 255
        ).astype(np.uint8)
        
        # Add tape lines
        tape_color = (200, 180, 140)
        tape_y = pkg_y + pkg_h // 2
        cv2.rectangle(image, (pkg_x, tape_y - 10), (pkg_x + pkg_w, tape_y + 10), tape_color, -1)
        
        # Add edge highlights
        edge_color = tuple(max(0, c - 30) for c in pkg_color)
        cv2.rectangle(image, (pkg_x, pkg_y), (pkg_x + pkg_w, pkg_y + pkg_h), edge_color, 2)
        
        # Store package bounds for damage placement
        image = np.ascontiguousarray(image)
        self._current_package = (pkg_x, pkg_y, pkg_w, pkg_h)
        
        return image
    
    def _add_damage(self, image: np.ndarray, damage_type: int) -> Tuple[float, float, float, float]:
        """
        Add damage to image and return bounding box.
        
        Returns:
            Normalized bbox (x_center, y_center, width, height)
        """
        pkg_x, pkg_y, pkg_w, pkg_h = self._current_package
        h, w = image.shape[:2]
        
        # Random damage position within package
        dmg_x = random.randint(pkg_x + 20, pkg_x + pkg_w - 50)
        dmg_y = random.randint(pkg_y + 20, pkg_y + pkg_h - 50)
        dmg_w = random.randint(30, min(80, pkg_w // 3))
        dmg_h = random.randint(30, min(80, pkg_h // 3))
        
        if damage_type == 0:  # structural_deformation (dent)
            self._draw_dent(image, dmg_x, dmg_y, dmg_w, dmg_h)
        elif damage_type == 1:  # surface_breach (tear)
            self._draw_tear(image, dmg_x, dmg_y, dmg_w, dmg_h)
        elif damage_type == 2:  # contamination_stain
            self._draw_stain(image, dmg_x, dmg_y, dmg_w, dmg_h)
        elif damage_type == 3:  # compression_damage
            self._draw_crush(image, dmg_x, dmg_y, dmg_w, dmg_h)
        elif damage_type == 4:  # tape_seal_damage
            self._draw_tape_damage(image, dmg_x, dmg_y, dmg_w, dmg_h)
        
        # Return normalized bbox
        x_center = (dmg_x + dmg_w / 2) / w
        y_center = (dmg_y + dmg_h / 2) / h
        norm_w = dmg_w / w
        norm_h = dmg_h / h
        
        return (x_center, y_center, norm_w, norm_h)
    
    def _draw_dent(self, image: np.ndarray, x: int, y: int, w: int, h: int):
        """Draw a dent (darker ellipse with highlight edge)."""
        center = (x + w // 2, y + h // 2)
        axes = (w // 2, h // 2)
        
        # Darker center (shadow)
        overlay = image.copy()
        cv2.ellipse(overlay, center, axes, 0, 0, 360, (50, 45, 40), -1)
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        
        # Highlight edge (top)
        cv2.ellipse(image, center, axes, 0, 200, 340, (180, 170, 160), 2)
    
    def _draw_tear(self, image: np.ndarray, x: int, y: int, w: int, h: int):
        """Draw a tear (jagged line with dark interior)."""
        # Dark interior color (exposed inner material)
        inner_color = (80, 70, 60)
        
        # Create jagged polygon
        points = [
            (x, y + h // 2),
            (x + w // 4, y + random.randint(0, h // 3)),
            (x + w // 2, y + h // 2 + random.randint(-10, 10)),
            (x + 3 * w // 4, y + random.randint(2 * h // 3, h)),
            (x + w, y + h // 2),
        ]
        points = np.array(points, dtype=np.int32)
        
        cv2.fillPoly(image, [points], inner_color)
        cv2.polylines(image, [points], False, (40, 35, 30), 2)
    
    def _draw_stain(self, image: np.ndarray, x: int, y: int, w: int, h: int):
        """Draw a water/contamination stain."""
        center = (x + w // 2, y + h // 2)
        
        # Stain color (brownish water stain)
        stain_color = random.choice([
            (100, 90, 70),   # Brown water stain
            (80, 80, 90),    # Gray stain
            (90, 100, 80),   # Greenish (mold)
        ])
        
        # Draw irregular stain shape
        overlay = image.copy()
        
        # Multiple overlapping ellipses for irregular shape
        for _ in range(3):
            offset = (random.randint(-10, 10), random.randint(-10, 10))
            size_var = (random.randint(w // 3, w // 2), random.randint(h // 3, h // 2))
            cv2.ellipse(
                overlay,
                (center[0] + offset[0], center[1] + offset[1]),
                size_var, 0, 0, 360, stain_color, -1
            )
        
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        # Add tide mark edge
        cv2.ellipse(image, center, (w // 2 - 5, h // 2 - 5), 0, 0, 360, 
                   (stain_color[0] - 20, stain_color[1] - 20, stain_color[2] - 20), 1)
    
    def _draw_crush(self, image: np.ndarray, x: int, y: int, w: int, h: int):
        """Draw compression damage (creased lines)."""
        # Multiple crease lines
        for i in range(3, 6):
            y_offset = y + (h * i) // 6
            
            # Wavy crease line
            pts = []
            for px in range(x, x + w, 5):
                py = y_offset + random.randint(-3, 3)
                pts.append((px, py))
            
            pts = np.array(pts, dtype=np.int32)
            
            # Shadow line
            cv2.polylines(image, [pts], False, (60, 55, 50), 2)
            
            # Highlight
            pts_highlight = pts.copy()
            pts_highlight[:, 1] -= 2
            cv2.polylines(image, [pts_highlight], False, (180, 170, 160), 1)
    
    def _draw_tape_damage(self, image: np.ndarray, x: int, y: int, w: int, h: int):
        """Draw tape seal damage."""
        # Torn/peeling tape
        tape_color = (200, 180, 140)
        exposed_color = (100, 90, 80)
        
        # Tape base
        cv2.rectangle(image, (x, y), (x + w, y + h // 3), tape_color, -1)
        
        # Torn section
        pts = np.array([
            (x + w // 3, y),
            (x + w // 2, y + h // 2),
            (x + 2 * w // 3, y + h // 3),
            (x + w, y),
        ], dtype=np.int32)
        cv2.fillPoly(image, [pts], exposed_color)
        
        # Tape edge
        cv2.polylines(image, [pts], False, (150, 130, 100), 2)
    
    def _save_sample(self, sample: Dict[str, Any], split: str):
        """Save image and annotation."""
        name = sample["name"]
        image = sample["image"]
        annotations = sample["annotations"]
        
        # Save image
        img_path = self.output_dir / "images" / split / f"{name}.jpg"
        cv2.imwrite(str(img_path), image)
        
        # Save label (YOLO format)
        label_path = self.output_dir / "labels" / split / f"{name}.txt"
        
        with open(label_path, "w") as f:
            for ann in annotations:
                class_id = ann["class_id"]
                x_center, y_center, w, h = ann["bbox"]
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def generate_sample_dataset(
    output_dir: str = "data",
    num_clean: int = 50,
    num_damaged: int = 50
):
    """
    Generate a sample dataset for testing.
    
    Args:
        output_dir: Output directory
        num_clean: Number of clean images
        num_damaged: Number of damaged images
    """
    generator = SampleDatasetGenerator(
        output_dir=output_dir,
        num_clean=num_clean,
        num_damaged=num_damaged
    )
    
    return generator.generate()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample dataset")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--clean", type=int, default=50, help="Number of clean images")
    parser.add_argument("--damaged", type=int, default=50, help="Number of damaged images")
    
    args = parser.parse_args()
    
    result = generate_sample_dataset(
        output_dir=args.output,
        num_clean=args.clean,
        num_damaged=args.damaged
    )
    
    print(f"\nDataset generated successfully!")
    print(f"Training samples: {result['train']}")
    print(f"Validation samples: {result['val']}")
