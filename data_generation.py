import os
import numpy as np
import cv2
import os
import urllib.request
import time
import openai


def generate_backgrounds(subject):
    """
    Generates ideas for backgrounds for images. 

    Args:
        None

    Returns:
        A list of strings of ideas for image backgrounds. 
    """
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Describe some places to use as a background in an image of a {subject} as a comma delimited list:",
        # 20 tokens is enough to produce 8 to 10 ideas.
        max_tokens=20,
        # temperature of 0.5 ensures we don't get the same ideas each time.
        temperature=0.5,
        n=1
    )
    # Parse the response
    text = response["choices"][0]["text"].strip('\n').lower()
    # Return a list of subjects
    return str.split(text, sep=', ', maxsplit=-1)


def mask_image(rgb_img, seg_mask):
    """
    Creates a mask from a image with a greenscreen backdrop.

    Args:
        rgb_img (str): The filepath of the input image which should contain
        a subject in front of a greenscreen background.
        seg_mask (str): The filepath to save the segmentation mask to. 

    Returns:
        None. Overwrites the original imaage with a new image that has the fourth channel 
        added and saves a new segmentation mask.
    """
    # Read the image.
    img = cv2.imread(rgb_img)
    # Convert from RGB to LAB.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Threshold the image.
    a_channel = lab[:, :, 1]
    th = cv2.threshold(a_channel, 127, 255,
                       cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # Replace "greenscreen".
    masked = cv2.bitwise_and(img, img, mask=th)
    masked[th == 0] = (255, 255, 255)
    # Add alpha channel to the original image.
    masked = np.dstack((masked, th))
    # Save the original image, now with a fourth alpha channel.
    cv2.imwrite(rgb_img, masked)
    # Save the segmentation mask.
    cv2.imwrite(seg_mask, th)


def generate_examples(subject, number_of_examples=3, original_dir='original_images', image_dir='images', mask_dir='masks'):
    """
    Creates images of an input subject with segmentation masks.

    Args:
        subject (str): A short description of the subject for the images.
        number_of_examples (int): Default=3. The number of images and masks to generate.
        image_save_dir (str): The directory where images should be saved to.
        mask_save_dir (str): The directory where segmentation masks should be saved to.

    Returns:
        A list of filepaths pointing to the newly-generated images. 
    """
    if not os.path.exists(original_dir):
        os.mkdir(original_dir)
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    # Generate ideas for background subjects.
    backgrounds = generate_backgrounds(subject)
    # Instantiate a list to store the filepaths of the images created.
    generated_images = []
    generated_masks = []

    for _ in range(number_of_examples):
        # Get string of current time to use for unique file name for the generated images.
        timestr = time.strftime("%Y%m%d-%H%M%S")
        # Generate an image of the subject on a "greenscreen".
        response = openai.Image.create(
            prompt=f"a photo realistic image of a {subject} on top of a greenscreen background",
            n=number_of_examples,
            size="1024x1024"
        )

        # Save the image locally.
        original_file_path = os.path.join(
            original_dir, f"{subject}_{timestr}.png")
        urllib.request.urlretrieve(
            response['data'][0]["url"], original_file_path)
        segmentation_file_path = os.path.join(
            mask_dir, f"{subject}_{timestr}.png")

        # Create segmentation mask and add fourth channel to the image.
        mask_image(original_file_path, segmentation_file_path)

        # Choose a background to use.
        background = np.random.choice(backgrounds, size=1)[0]

        # Generate an image of the subject on a new background.
        response = openai.Image.create_edit(
            image=open(original_file_path, "rb"),
            prompt=f"a {subject} in front of a {background}",
            n=1,
            size="1024x1024")

        # Save the image locally.
        image_file_path = os.path.join(image_dir, f"{subject}_{timestr}.png")
        urllib.request.urlretrieve(response["data"][0]["url"], image_file_path)
        # Append the file name to the list to return.
        generated_images.append(image_file_path)
        generated_masks.append(segmentation_file_path)
    # Return a list of the images that were generated.
    return (generated_images, generated_masks)
