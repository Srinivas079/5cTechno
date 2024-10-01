Image Data Loader with Batch Processing for Brain MRI Segmentation
This code implements a memory-efficient image data loading mechanism designed for training deep learning models in the context of brain MRI metastasis segmentation. The dataset comprises TIFF images and their corresponding mask images, organized in a directory structure that includes multiple subfolders.

Key Features:
Batch Processing: The load_data function processes images and masks in batches, significantly reducing memory consumption during data loading. This is especially beneficial when working with large datasets or high-resolution images.

Dynamic Image Resizing: Each image and mask is resized to a uniform shape of (256, 256) pixels using bilinear interpolation with anti-aliasing. This ensures that the model receives inputs of consistent dimensions, which is essential for training convolutional neural networks.

Directory Traversal: The code traverses the specified directory and its subfolders to identify and load all TIFF images. It distinguishes between standard images and mask images based on the presence of "_mask" in the filename, ensuring that corresponding masks are paired with their respective images.

Numpy Array Handling: Images and masks are collected into lists and converted into NumPy arrays after processing each batch. This approach allows for efficient handling of data in subsequent steps of model training and evaluation.

Training and Validation Split: After loading the complete dataset, it splits the data into training and validation sets using an 80-20 ratio, preparing the data for model training.

Usage:
This implementation is designed to be integrated into a Jupyter Notebook environment or any Python script where deep learning models for image segmentation are being developed. It ensures that the data loading process is both efficient and scalable, allowing for seamless transitions between data preprocessing and model training phases.
