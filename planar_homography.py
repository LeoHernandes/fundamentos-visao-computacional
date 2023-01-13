import numpy as np
import cv2 as cv
import cvui

# Photos paths:
BEACH_PERSPECTIVE_1     = "./photos/foto1_cap1.jpg"
BEACH_PERSPECTIVE_2     = "./photos/foto1_cap2.jpg"
BEACH_ORIGINAL          = "./photos/foto1_gt.jpg"
SUNSET_PERSPECTIVE_1    = "./photos/foto2_cap1.jpg"
SUNSET_PERSPECTIVE_2    = "./photos/foto2_cap2.jpg"
SUNSET_ORIGINAL         = "./photos/foto2_gt.jpg"


def on_mouse_click(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONUP and len(image_points) < 4:
        # Save previous image state on cache
        image_state = params[0].copy()
        image_cache.append(image_state)
        # Save marked point
        image_points.append([x, y])
        # Draw a circle to mark the point
        cv.circle(params[0], (x, y), radius = 1, color = (0, 0, 255), thickness = 5)


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


def load_resized_image(image_path):
    image = cv.imread(image_path)
    
    # Fits allways on a 1280 x 720 px screen
    if image.shape[0] >= image.shape[1]:
        return resize_with_aspect_ratio(image, desired_height = 720)
    else:
        return resize_with_aspect_ratio(image, desired_width = 720)


def sort_image_points(image_points):
    # Creates a new list with points ordered by clockwise starting from top left corner
    sorted_points = []
    
    sorted_by_x = sorted(image_points , key=lambda k: k[0])
   
    if sorted_by_x[0][1] < sorted_by_x[1][1]:
        sorted_points.append(sorted_by_x[0])
        del sorted_by_x[0]
    else:
        sorted_points.append(sorted_by_x[1])
        del sorted_by_x[1]
    
    if sorted_by_x[1][1] < sorted_by_x[2][1]:
        sorted_points.append(sorted_by_x[1])
        del sorted_by_x[1]
    else:
        sorted_points.append(sorted_by_x[2])
        del sorted_by_x[2]
    
    sorted_points.append(sorted_by_x[1])
    sorted_points.append(sorted_by_x[0])

    return sorted_points


def smallest_side(quadrilateral_points):
    left_side = quadrilateral_points[3][1] - quadrilateral_points[0][1]
    right_side = quadrilateral_points[2][1] - quadrilateral_points[1][1]
    top_side = quadrilateral_points[1][0] - quadrilateral_points[0][0]
    bottom_side = quadrilateral_points[2][0] - quadrilateral_points[3][0]
    
    distances = [left_side, right_side, top_side, bottom_side]
    
    min_distance = min(distances)
    index = distances.index(min_distance)
    if index < 2:
        return min_distance, "vertical"
    else:
        return min_distance, "horizontal"


def calculate_desired_points(original_dimensions, marked_points):
    image_width = original_dimensions[0]
    image_height = original_dimensions[1]
    
    destine_points = []
    
    # Gets orientation of the smallest size of marked region 
    distance, orientation = smallest_side(marked_points)
    
    # Arbitrarily fixes the top left corner point
    fixed_point = marked_points[0]
    destine_points.append(fixed_point)
    
    if orientation == "horizontal":
        # Get the correct image ratio based on the smallest side
        new_height = int((distance * image_height)/float(image_width))
        # Construct a retangular shape with this ratio and fixed on the top left corner point
        destine_points.append([fixed_point[0] + distance, fixed_point[1]])
        destine_points.append([fixed_point[0] + distance, fixed_point[1] + new_height])
        destine_points.append([fixed_point[0], fixed_point[1] + new_height])
        return destine_points        

    if orientation == "vertical":
        new_width = int((distance * image_width)/float(image_height))
        destine_points.append([fixed_point[0] + new_width, fixed_point[1]])
        destine_points.append([fixed_point[0] + new_width, fixed_point[1] + distance])
        destine_points.append([fixed_point[0], fixed_point[1] + distance])
        return destine_points  


def apply_homography(image, image_points, original_dimensions):
    sorted_points = sort_image_points(image_points)
    desired_image_points = calculate_desired_points(original_dimensions, sorted_points)
    
    homography_matrix, status = cv.findHomography(np.array(sorted_points), np.array(desired_image_points))
    return cv.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0])), desired_image_points


# ---------------------- MAIN CODE ---------------------- # 
def get_image_proportions(image_path):
    original_image = cv.imread(image_path)
    return (original_image.shape[1], original_image.shape[0])

if __name__ == "__main__":
    image_points = []                   # Marked points on image
    image_cache = []                    # Cache of marked points, used on "undo" button
    perspective_image = None            # Image to be processed
    perspective_image_copy = None       # Image copy to draw points
    original_image_dimensions = None    # Original image width and height (in this order) to make correct proportions
    crop_points = None                  # Stores the destine points of the homography operation 
    is_homography_applied = False       # Controls the state of program

    frame = np.zeros((420, 200, 3), dtype = "uint8")
    frame.fill(32)

    cvui.init("Menu")

    cv.namedWindow('image')

    # Program loop ----------------------
    while(True):
        
        # Interface buttons ----------------------
        if cvui.button(frame, 20, 20, 160, 40, "Load beach image 1"):
            perspective_image = load_resized_image(BEACH_PERSPECTIVE_1)
            perspective_image_copy = perspective_image.copy()
            original_image_dimensions = get_image_proportions(BEACH_ORIGINAL)
            image_points = []
            is_homography_applied = False

        if cvui.button(frame, 20, 70, 160, 40, "Load beach image 2"):
            perspective_image = load_resized_image(BEACH_PERSPECTIVE_2)
            perspective_image_copy = perspective_image.copy()
            original_image_dimensions = get_image_proportions(BEACH_ORIGINAL)
            image_points = []
            is_homography_applied = False

        if cvui.button(frame, 20, 120, 160, 40, "Load sunset image 1"):
            perspective_image = load_resized_image(SUNSET_PERSPECTIVE_1)
            perspective_image_copy = perspective_image.copy()
            original_image_dimensions = get_image_proportions(SUNSET_ORIGINAL)
            image_points = []
            is_homography_applied = False

        if cvui.button(frame, 20, 170, 160, 40, "Load sunset image 2"):
            perspective_image = load_resized_image(SUNSET_PERSPECTIVE_2)
            perspective_image_copy = perspective_image.copy()
            original_image_dimensions = get_image_proportions(SUNSET_ORIGINAL)
            image_points = []
            is_homography_applied = False

        if cvui.button(frame, 20, 240, 160, 40, "Undo marked point") and len(image_cache) > 0 and not is_homography_applied:
            perspective_image_copy = image_cache.pop()
            image_points.pop()
        
        if cvui.button(frame, 20, 300, 160, 40, "Apply homography") and len(image_points) == 4:
            perspective_image_copy, crop_points = apply_homography(perspective_image, image_points, original_image_dimensions)
            is_homography_applied = True
        
        if cvui.button(frame, 20, 360, 160, 40, "Crop image") and is_homography_applied:
            perspective_image_copy = perspective_image_copy[crop_points[0][1] : crop_points[3][1],
                                                            crop_points[0][0] : crop_points[1][0]]
        
        # Create windows ----------------------
        if perspective_image_copy is not None:
            cv.setMouseCallback('image', on_mouse_click, [perspective_image_copy])
            cv.imshow('image', perspective_image_copy)
        
        cvui.update()
        cv.imshow("Menu", frame)
        
        # Exit condition ----------------------
        if cv.waitKey(20) & 0xFF == 27:
            break

    cv.destroyAllWindows()