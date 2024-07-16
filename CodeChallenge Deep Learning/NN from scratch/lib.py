import cv2

def preprocess_image(image_path):
    """
    Preprocess an image: resize, normalize, and reshape.

    Parameters:
    - image_path (str): Path to the input image file.

    Returns:
    - image (numpy.ndarray): Preprocessed image array.
    """
    try:
        # Read the image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize to match MNIST digit size (28x28)
        image = cv2.resize(image, (28, 28))
        
        # Normalize pixel values to [0, 1]
        image_normalized = image.astype('float32') / 255.0
        
        return  image_normalized
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None
    
import glob

def Image(image_dir="./numbers"):

  """
  Loads and preprocesses images from the specified directory.

  Args:
    image_dir: Path to the directory containing image files.

  Returns:
    A list of preprocessed image arrays.
  """

  image_paths = glob.glob(f"{image_dir}/*.jpg")
  images = [preprocess_image(img) for img in image_paths]
  return images