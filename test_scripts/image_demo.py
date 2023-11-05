import cv2


def process_and_show_image(image_path):
    """Reads an image from a file, converts it to grayscale, displays the
    original and grayscale images, and waits for user input.

    Parameters
    ----------
    image_path : str
        Path to the image file to load and process

    Returns
    -------
    None

    Examples
    --------
    >>> process_and_show_image('example.jpg')

    This will load example.jpg, display the color and grayscale
    versions side by side, and wait for user input before closing the windows.
    """
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to read the image.")
        return

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Original Image", image)
    cv2.imshow("Grayscale Image", grayscale_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
