from PIL import Image, ImageDraw
import random
import os
from typing import Tuple, Optional
import json

class JellyBeanGenerator:
    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        background_color: Tuple[int, int, int] = (240, 240, 240),
        min_jellybeans: int = 100,
        max_jellybeans: int = 500,
        output_dir: str = "jellybean_images"
    ):
        """
        Generate synthetic jellybean images with random counts, colors, and positions.
        
        Args:
            image_size: (width, height) of output images
            background_color: RGB color for background
            min_jellybeans: Minimum number of jellybeans per image
            max_jellybeans: Maximum number of jellybeans per image
            output_dir: Directory to save generated images
        """
        self.image_size = image_size
        self.background_color = background_color
        self.min_jellybeans = min_jellybeans
        self.max_jellybeans = max_jellybeans
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Predefined jellybean colors (can be customized)
        self.jellybean_colors = [
            (255, 50, 50),    # Red
            (50, 255, 50),    # Green
            (50, 50, 255),    # Blue
            (255, 255, 50),   # Yellow
            (255, 50, 255),   # Magenta
            (50, 255, 255),   # Cyan
            (255, 150, 50),   # Orange
            (200, 50, 255),   # Purple
        ]

    def _generate_jellybean_shape(
        self,
        draw: ImageDraw.Draw,
        center: Tuple[int, int],
        color: Tuple[int, int, int],
        size_variation: float = 0.2
    ) -> None:
        """Draw a single jellybean (ellipse with highlights)"""
        width, height = self.image_size
        bean_width = random.randint(int(width*0.015), int(width*0.035))
        bean_height = random.randint(int(height*0.02), int(height*0.045))
        
        # Apply size variation
        bean_width = int(bean_width * (1 + random.uniform(-size_variation, size_variation)))
        bean_height = int(bean_height * (1 + random.uniform(-size_variation, size_variation)))
        
        # Draw main jellybean body (ellipse)
        bbox = [
            (center[0] - bean_width, center[1] - bean_height),
            (center[0] + bean_width, center[1] + bean_height)
        ]
        draw.ellipse(bbox, fill=color, outline=(0, 0, 0, 50), width=1)
        
        # Add highlight for 3D effect
        highlight_color = (
            min(color[0] + 50, 255),
            min(color[1] + 50, 255),
            min(color[2] + 50, 255)
        )
        highlight_bbox = [
            (center[0] - bean_width//2, center[1] - bean_height//2),
            (center[0] - bean_width//4, center[1] - bean_height//4)
        ]
        draw.ellipse(highlight_bbox, fill=highlight_color)

    def generate_image(
        self,
        num_jellybeans: Optional[int] = None,
        filename: Optional[str] = None
    ) -> Tuple[Image.Image, int]:
        """
        Generate a single jellybean image.
        
        Args:
            num_jellybeans: Fixed number of jellybeans (random if None)
            filename: If provided, saves to this filename
            
        Returns:
            (PIL.Image, count) tuple
        """
        if num_jellybeans is None:
            num_jellybeans = random.randint(self.min_jellybeans, self.max_jellybeans)
        
        # Create blank image
        img = Image.new("RGB", self.image_size, self.background_color)
        draw = ImageDraw.Draw(img)
        
        # Generate jellybeans
        for _ in range(num_jellybeans):
            # Random position (with border margin)
            margin = 50
            x = random.randint(margin, self.image_size[0] - margin)
            y = random.randint(margin, self.image_size[1] - margin)
            
            # Random color (or pick from predefined)
            if random.random() > 0.2:  # 80% chance to use predefined colors
                color = random.choice(self.jellybean_colors)
            else:  # 20% chance for random color
                color = (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255)
                )
            
            self._generate_jellybean_shape(draw, (x, y), color)
        
        # Save if filename provided
        if filename:
            img.save(os.path.join(self.output_dir, filename))
        
        return img, num_jellybeans

    def generate_dataset(
        self,
        num_images: int,
        prefix: str = "jellybean"
    ) -> None:
        """
        Generate multiple jellybean images with annotation file.
        
        Args:
            num_images: Number of images to generate
            prefix: Filename prefix
        """
        annotation_fp = os.path.join(self.output_dir, "annotations.json")
        annotations = dict()
        
        for i in range(80 * num_images // 100):
            filename = f"{prefix}_{i:03d}.png"
            _, count = self.generate_image(filename=filename)
            annotations[filename] = count

        for i in range(20 * num_images // 100):
            filename = f"{prefix}_{(i + (80 * num_images // 100)):03d}.png"
            _, count = self.generate_image(num_jellybeans=0, filename=filename)
            annotations[filename] = count
        
        with open(annotation_fp, "w") as file:
            json.dump(annotations, file)
    
    def randomize_dataset(self, num_images, dataset):
        for name in list(dataset.annotations.keys())[0:min(num_images, len(dataset.annotations.keys()))]:
            _, count = self.generate_image(filename=name)
            dataset.annotations[name] = count

        annotation_fp = os.path.join(self.output_dir, "annotations.json")
        
        with open(annotation_fp, "w") as file:
            json.dump(dataset.annotations, file)  

# Example usage
if __name__ == "__main__":
    generator = JellyBeanGenerator(
        image_size=(512, 512),
        min_jellybeans=100,
        max_jellybeans=500,
        output_dir="jellybeans_dataset"
    )

    # Generate full dataset (100 images)
    generator.generate_dataset(1000)