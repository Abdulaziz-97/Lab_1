import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class ImageProcessor:
    def __init__(self):
        """Initialize the Image Processor"""
        self.original_image = None
        self.image_path = None
    
    def load_image(self, image_path):
        """
        Load a grayscale image and display its properties
        
        Args:
            image_path (str): Path to the image file
        """
        # Load image in grayscale mode
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.image_path = image_path
        
        if self.original_image is None:
            print(f"Error: Could not load image from {image_path}")
            return False
        
        # Display image properties
        height, width = self.original_image.shape
        data_type = self.original_image.dtype
        bits_per_pixel = 8 if data_type == np.uint8 else 16 if data_type == np.uint16 else 32
        
        print("=== IMAGE PROPERTIES ===")
        print(f"Image loaded: {os.path.basename(image_path)}")
        print(f"Dimensions (Width x Height): {width} x {height} pixels")
        print(f"Total pixels: {width * height:,}")
        print(f"Data type: {data_type}")
        print(f"Bits per pixel: {bits_per_pixel}")
        print(f"Number of rows: {height}")
        print(f"Number of columns: {width}")
        print(f"Pixel value range: {self.original_image.min()} - {self.original_image.max()}")
        print("=" * 30)
        
        return True
    
    def display_image(self, image, title="Image", cmap='gray'):
        """
        Display a single image
        
        Args:
            image: Image array to display
            title (str): Title for the image
            cmap (str): Colormap for display
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    def image_negation(self):
        """
        Implement image negation operation
        Formula: g(x,y) = L - 1 - f(x,y)
        Where L is the number of gray levels (256 for 8-bit images)
        """
        if self.original_image is None:
            print("Error: No image loaded. Please load an image first.")
            return None
        
        # Apply negation formula: g(x,y) = L - 1 - f(x,y)
        L = 256  # Number of gray levels for 8-bit image
        negative_image = L - 1 - self.original_image
        
        # Display original and negative images side by side
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(self.original_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(negative_image, cmap='gray')
        plt.title('Negative Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("=== IMAGE NEGATION ANALYSIS ===")
        print("Visual Effect: Image negation reverses the intensity values.")
        print("- Light areas become dark")
        print("- Dark areas become light")
        print("- Creates a 'photographic negative' effect")
        print("\nCommon applications include:")
        print("- Medical imaging (X-rays, where bones appear white)")
        print("- Enhancing dark features in bright backgrounds")
        print("- Preprocessing for certain computer vision algorithms")
        print("- Creating artistic effects")
        print("=" * 40)
        
        return negative_image
    
    def binary_thresholding(self, threshold_values=[64, 128, 192]):
        """
        Implement binary thresholding operation
        Formula: g(x,y) = 255 if f(x,y) > T, else 0
        
        Args:
            threshold_values (list): List of threshold values to test
        """
        if self.original_image is None:
            print("Error: No image loaded. Please load an image first.")
            return None
        
        # Create subplots for original and thresholded images
        num_thresholds = len(threshold_values)
        plt.figure(figsize=(5 * (num_thresholds + 1), 6))
        
        # Display original image
        plt.subplot(1, num_thresholds + 1, 1)
        plt.imshow(self.original_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        thresholded_images = {}
        
        # Apply thresholding for each threshold value
        for i, T in enumerate(threshold_values):
            # Apply binary thresholding
            # g(x,y) = 255 if f(x,y) > T, else 0
            binary_image = np.where(self.original_image > T, 255, 0).astype(np.uint8)
            thresholded_images[T] = binary_image
            
            # Display thresholded image
            plt.subplot(1, num_thresholds + 1, i + 2)
            plt.imshow(binary_image, cmap='gray')
            plt.title(f'Threshold = {T}')
            plt.axis('off')
            
            # Calculate and display statistics
            white_pixels = np.sum(binary_image == 255)
            black_pixels = np.sum(binary_image == 0)
            total_pixels = binary_image.size
            white_percentage = (white_pixels / total_pixels) * 100
            
            print(f"Threshold T = {T}:")
            print(f"  White pixels: {white_pixels:,} ({white_percentage:.1f}%)")
            print(f"  Black pixels: {black_pixels:,} ({100-white_percentage:.1f}%)")
        
        plt.tight_layout()
        plt.show()
        
        print("\n=== THRESHOLDING ANALYSIS ===")
        print("Effect of changing threshold value T:")
        print("- Lower T values: More pixels become white (255)")
        print("  -> More liberal thresholding, preserves more details")
        print("- Higher T values: Fewer pixels become white (255)")
        print("  -> More conservative thresholding, only brightest regions remain")
        print("- T = 128 (middle value): Balanced segmentation")
        print("\nApplications:")
        print("- Object segmentation from background")
        print("- Text extraction from documents")
        print("- Creating binary masks for further processing")
        print("- Preprocessing for shape analysis")
        print("=" * 40)
        
        return thresholded_images
    
    def demonstrate_all_operations(self, image_path):
        """
        Demonstrate all lab operations in sequence
        
        Args:
            image_path (str): Path to the input image
        """
        print("\n" + "=" * 60)
        print("PART 1: IMAGE LOADING AND ANALYSIS")
        print("=" * 60)
        
        # Display the original image
        self.display_image(self.original_image, "Original Loaded Image")
        
        # 2. Image negation
        print("\n" + "=" * 60)
        print("PART 2: IMAGE NEGATION")
        print("=" * 60)
        negative_img = self.image_negation()
        
        # 3. Binary thresholding
        print("\n" + "=" * 60)
        print("PART 3: BINARY THRESHOLDING")
        print("=" * 60)
        thresholded_imgs = self.binary_thresholding([64, 128, 192])
        
        print("\n" + "=" * 60)
        print("LAB 1 COMPLETED SUCCESSFULLY!")
        print("=" * 60)

def main():
    """
    Main function to run the lab demonstration
    """
    processor = ImageProcessor()
    
    # Try to find a sample image in common locations
    sample_images = [
        "test.jpg"
    ]
    
    image_found = False
    for img_path in sample_images:
        if os.path.exists(img_path):
            print(f"Trying to load: {img_path}")
            if processor.load_image(img_path):
                processor.demonstrate_all_operations(img_path)
                image_found = True
                break
            else:
                print(f"Failed to load {img_path}, trying next image...")
                continue
    
    if not image_found:
        print("No sample image found in the current directory.")
        print("Please place a grayscale image file (jpg, png) in the same directory")
        print("with one of these names:", ", ".join(sample_images))
        print("\nAlternatively, you can manually specify the image path:")
        print("processor = ImageProcessor()")
        print("processor.demonstrate_all_operations('your_image_path.jpg')")
        
        # Create a synthetic image for demonstration if no real image is available
        print("\nCreating a synthetic image for demonstration...")
        create_sample_image()
        processor.demonstrate_all_operations("synthetic_sample.png")

def create_sample_image():
    """
    Create a synthetic sample image for demonstration purposes
    """
    # Create a synthetic image with various intensity regions
    img = np.zeros((200, 200), dtype=np.uint8)
    
    # Add different intensity regions
    img[50:100, 50:100] = 80    # Dark gray square
    img[25:75, 125:175] = 160   # Light gray square
    img[125:175, 25:75] = 200   # Very light gray square
    img[100:150, 100:150] = 40  # Very dark gray square
    
    # Add some noise
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Save the synthetic image
    cv2.imwrite("synthetic_sample.png", img)
    print("Created synthetic_sample.png for demonstration")

if __name__ == "__main__":
    main()
