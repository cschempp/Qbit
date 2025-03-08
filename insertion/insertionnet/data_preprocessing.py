import numpy as np
import cv2
import random
import h5py
import glob
import os
from tqdm import tqdm


def random_convolution(image):
    """
    Apply a random convolution kernel to the image.
    """
    kernel_size = np.random.choice([3, 5])
    kernel = np.random.rand(kernel_size, kernel_size)
    kernel /= np.sum(np.abs(kernel))
    augmented_image = cv2.filter2D(image, -1, kernel)
    return np.clip(augmented_image, 0, 255).astype(np.uint8)


def color_jitter(image):
    """
    Convert the image to HSV, add noise to HSV channels, and convert it back to RGB.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_jitter = np.random.uniform(-90, 90)
    saturation_jitter = np.random.uniform(0.5, 1.5)
    value_jitter = np.random.uniform(0.5, 1.5)
    hsv_image[..., 0] = (hsv_image[..., 0] + hue_jitter) % 180
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * saturation_jitter, 0, 255)
    hsv_image[..., 2] = np.clip(hsv_image[..., 2] * value_jitter, 0, 255)
    augmented_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return augmented_image


def grayscale(image):
    """
    Convert the image to grayscale.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)


def random_translation(image, frame_size):
    """
    Translate the image within a larger frame of size (frame_size, frame_size).
    """
    original_size = image.shape[0]
    frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
    x_offset = np.random.randint(0, frame_size - original_size + 1)
    y_offset = np.random.randint(0, frame_size - original_size + 1)
    frame[y_offset:y_offset + original_size, x_offset:x_offset + original_size] = image
    return frame


def random_cropping(image, crop_size):
    """
    Crop a random patch from the image.
    """
    h, w, _ = image.shape
    crop_h, crop_w = crop_size
    
    if crop_h > h or crop_w > w:
        raise ValueError("Crop size must be smaller than the original image size.")
    
    x_start = np.random.randint(0, w - crop_w + 1)
    y_start = np.random.randint(0, h - crop_h + 1)
    
    cropped_image = image[y_start:y_start + crop_h, x_start:x_start + crop_w]
    return cropped_image


def scale_force_torque_vector(force_torque_vector):
    """
    Augment a 1x6 force-torque vector by scaling it with a factor sampled from [0, 1].
    """
    scale_factor = np.random.uniform(0, 1)
    return force_torque_vector * scale_factor


def apply_augmentations(image):
    """
    Augmentation pipeline for images.
    """
    augmentations = [
        random_convolution,
        color_jitter,
        grayscale,
        lambda img: random_translation(img, frame_size=max(img.shape[:2]) + 15),
        lambda img: random_cropping(img, crop_size=[220, 220])
    ]
    aug_image = random.choice(augmentations)(image)
    return aug_image


def augment_dataset(input_images,
                    input_force_torque,
                    policy_actions,
                    output_hdf5,
                    target_size):
    """
    Augment the dataset by applying visual and force-torque augmentations.

    Args:
        input_images (list of np.array): List of input images.
        input_force_torque (list of np.array): List of 1x6 force-torque vectors corresponding to the images.
        output_hdf5 (str): Path to the output HDF5 file.
        target_size (int): Total number of augmented samples to generate.
    """
    total_images = len(input_images)
    augmentations_per_image = target_size // total_images

    with h5py.File(output_hdf5, "w") as hdf5_file:
        # Create datasets for images and force-torque vectors
        img_shape = input_images[0].shape
        ft_shape = (6,)
        action_shape = (5,)

        image_dataset = hdf5_file.create_dataset(
            "augmented_images", shape=(target_size,) + img_shape, dtype=np.uint8
        )
        force_torque_dataset = hdf5_file.create_dataset(
            "augmented_force_torque", shape=(target_size,) + ft_shape, dtype=np.float32
        )
        policy_actions_dataset = hdf5_file.create_dataset(
            "policy_actions", shape=(target_size,) + action_shape, dtype=np.float32
        )

        idx = 0
        for image, force_torque, action in tqdm(zip(input_images, input_force_torque, policy_actions), total=total_images):
            for _ in range(augmentations_per_image):
                aug_image = apply_augmentations(image)
                aug_image = cv2.resize(aug_image, input_images[0].shape[:2])
                
                # Apply force-torque augmentation
                aug_force_torque = scale_force_torque_vector(force_torque)

                # Save augmented data to HDF5
                if idx < target_size:
                    image_dataset[idx] = aug_image
                    force_torque_dataset[idx] = aug_force_torque
                    policy_actions_dataset[idx] = action * np.array([1, 1, np.pi/180, np.pi/180, np.pi/180])
                    idx += 1

                if idx >= target_size:
                    break

    print(f"Augmented dataset saved to {output_hdf5}")


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join("data", "**", "*.npy")))
    input_data = [[cv2.resize(np.load(file, allow_pickle=True)[2][:, 500:, :], (256, 256)), np.load(file, allow_pickle=True)[1], np.load(file, allow_pickle=True)[0]] for file in files]
    input_data = np.array(input_data, dtype=np.object_)
    input_images = input_data[:, 0]
    input_force_torque = input_data[:, 1]
    policy_actions = input_data[:, 2]

    output_hdf5_file = os.path.join("data", "augmented_dataset.h5")
    final_dataset_size = 100000

    augment_dataset(input_images, input_force_torque, policy_actions, output_hdf5_file, final_dataset_size)
