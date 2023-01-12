import numpy as np
import cv2 as cv
import cvui

# Photos paths:
BEACH_PERSPECTIVE_1     = "./photos/foto1_cap1.jpg"
BEACH_PERSPECTIVE_2     = "./photos/foto1_cap2.jpg"
BEACH_ORIGINAL          = "./photos/foto1_gt.jpg"
BEACH_ORIGINAL_HEIGHT   = 720
BEACH_ORIGINAL_WIDTH    = 960
SUNSET_PERSPECTIVE_1    = "./photos/foto2_cap1.jpg"
SUNSET_PERSPECTIVE_2    = "./photos/foto2_cap2.jpg"
SUNSET_ORIGINAL         = "./photos/foto2_gt.jpg"
SUNSET_ORIGINAL_HEIGHT   = 960      
SUNSET_ORIGINAL_WIDTH    = 720


def on_mouse_click(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONUP and len(image_points) < 4:
        # Save previous image state on cache
        image_state = perspective_image.copy()
        image_cache.append(image_state)
        # Save marked point
        image_points.append([x, y])
        # Draw a circle to mark the point
        cv.circle(perspective_image, (x, y), radius = 2, color = (0, 0, 255), thickness = 5)


def resize_with_aspect_ratio(image, desired_width = None, desired_height = None, inter = cv.INTER_AREA):
    dimension = None
    (original_height, original_width) = image.shape[:2]

    if desired_width is None and desired_height is None:
        return image
    
    if desired_width is None:
        ratio = desired_height / float(original_height)
        dimension = (int(original_width * ratio), desired_height)
    else:
        ratio = desired_width / float(original_width)
        dimension = (desired_width, int(original_height * ratio))

    return cv.resize(image, dimension, interpolation = inter)


def load_with_aspect_ratio(image_path, desired_height):
    iamge = cv.imread(image_path)
    return resize_with_aspect_ratio(iamge, desired_height = desired_height)


def apply_homography():
    image_points_array = np.array([image_points])
    desired_image_points = np.array([[0, 0],
                                     [desired_image_width, 0],
                                     [desired_image_width, desired_image_height], 
                                     [0, desired_image_height]])
    homography_matrix, status = cv.findHomography(image_points_array, desired_image_points)
    return cv.warpPerspective(perspective_image, homography_matrix, (perspective_image.shape[1], perspective_image.shape[0]))


# ---------------------- MAIN CODE ---------------------- # 
# Global variables
desired_image_width = None                                                          # Destine image width
desired_image_height = None                                                         # Destine image height
image_points = []                                                                   # Marked points on image
image_cache = []                                                                    # Cache of marked points, used on "undo" button
perspective_image = None                                                            # Image to be processed
is_homography_applied = False                                                       # Controls the state of program

frame = np.zeros((420, 200, 3), dtype = "uint8")
frame.fill(32)

cvui.init("Menu")

cv.namedWindow('image')
cv.setMouseCallback('image', on_mouse_click)

# Program loop ----------------------
while(True):
    
    # Interface buttons ----------------------
    if cvui.button(frame, 20, 20, 160, 40, "Load beach image 1"):
       perspective_image = load_with_aspect_ratio(BEACH_PERSPECTIVE_1, 720)
       desired_image_height = BEACH_ORIGINAL_HEIGHT
       desired_image_width = BEACH_ORIGINAL_WIDTH
       image_points = []
       is_homography_applied = False
    
    if cvui.button(frame, 20, 70, 160, 40, "Load beach image 2"):
       perspective_image = load_with_aspect_ratio(BEACH_PERSPECTIVE_2, 720)
       desired_image_height = BEACH_ORIGINAL_HEIGHT
       desired_image_width = BEACH_ORIGINAL_WIDTH
       image_points = []
       is_homography_applied = False
       
    if cvui.button(frame, 20, 120, 160, 40, "Load sunset image 1"):
       perspective_image = load_with_aspect_ratio(SUNSET_PERSPECTIVE_1, 720)
       desired_image_height = SUNSET_ORIGINAL_HEIGHT
       desired_image_width = SUNSET_ORIGINAL_WIDTH
       image_points = []
       is_homography_applied = False
    
    if cvui.button(frame, 20, 170, 160, 40, "Load sunset image 2"):
       perspective_image = load_with_aspect_ratio(SUNSET_PERSPECTIVE_2, 720)
       desired_image_height = SUNSET_ORIGINAL_HEIGHT
       desired_image_width = SUNSET_ORIGINAL_WIDTH
       image_points = []
       is_homography_applied = False
       
    if cvui.button(frame, 20, 240, 160, 40, "Undo marked point") and len(image_cache) > 0 and not is_homography_applied:
       perspective_image = image_cache.pop()
       image_points.pop()
    
    if cvui.button(frame, 20, 300, 160, 40, "Apply homography") and len(image_points) == 4:
       perspective_image = apply_homography()
       is_homography_applied = True
       
    if is_homography_applied:
        if cvui.button(frame, 20, 360, 160, 40, "Crop image"):
            perspective_image = perspective_image[0 : desired_image_height, 0 : desired_image_width]
    
    # Create windows ----------------------
    if perspective_image is not None:
        cv.imshow('image', perspective_image)
    
    cvui.update()
    cv.imshow("Menu", frame)
    
    # Exit condition ----------------------
    if cv.waitKey(20) & 0xFF == 27:
        break

cv.destroyAllWindows()