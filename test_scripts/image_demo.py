import cv2

def process_and_show_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to read the image.")
        return

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Original Image", image)
    cv2.imshow("Grayscale Image", grayscale_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
