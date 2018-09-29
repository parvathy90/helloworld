
# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Path to the pen training dataset")
ap.add_argument("-t", "--test", required=True, help="Path to the test dataset")
args = vars(ap.parse_args())

# initialize the data matrix and labels
print ("[INFO] extracting features...")
data = []
labels = []
# loop over the image paths in the training set
for imagePath in paths.list_images(args["training"]):
	# extract the type of pen
	make = imagePath.split("/")[-1]
 
	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edged = imutils.auto_canny(gray)
 
	# find contours in the edge map, keeping only the largest one which
	# is presmumed to be the pen
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	c = max(cnts, key=cv2.contourArea)
 
	# extract the type of pen and resize it to a canonical width
	# and height
	(x, y, w, h) = cv2.boundingRect(c)
	logo = gray[y:y + h, x:x + w]
	logo = cv2.resize(logo, (128, 128))
 
	# extract Histogram of Oriented Gradients from the pic of pen
	H = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
 
	# update the data and labels
	data.append(H)
	labels.append(make)
# "train" the nearest neighbors classifier
print("[INFO] training classifier...")
#model = KNeighborsClassifier(n_neighbors=4)
#model=DecisionTreeClassifier(max_depth=18)
model=SVC(C=10000,kernel="linear",gamma=0.00001)
model.fit(data, labels)
print("[INFO] evaluating...")
# loop over the test dataset
for (i, imagePath) in enumerate(paths.list_images(args["test"])):
	# load the test image, convert it to grayscale, and resize it to
	# the canonical size
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	logo = cv2.resize(gray, (128, 128))
	hog = cv2.HOGDescriptor() 
	# extract Histogram of Oriented Gradients from the test image and
	# predict the type of the pen
	(H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualise=True)
	pred = model.predict(H.reshape(1, -1))[0]
 
	# visualize the HOG image
	hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
	hogImage = hogImage.astype("uint8")
	#cv2.imshow("HOG Image #{}".format(i + 1), hogImage)
	print(pred.title())
	# draw the prediction on the test image and display it
	#cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		#(0, 255, 0), 3)
	#cv2.imshow("Test Image #{}".format(i + 1), image)
	#cv2.waitKey(0)

cap = cv2.VideoCapture(0)
dim = 128 # For HOG

while True:
    # Capture the frame
    ret, frame = cap.read()

    # Show the image on the screen
    cv2.imshow('Webcam', frame)

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the image into a HOG descriptor
    gray = cv2.resize(gray, (dim, dim), interpolation = cv2.INTER_AREA)
    (H, hogImage) = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualise=True)
    pred = model.predict(H.reshape(1, -1))[0]
    #features = hog.compute(gray)
    #features = features.T # Transpose so that the feature is in a single row
	"""
	destination_image = cv2.absdiff(image1, image2)
	def preprocess_image(image):
    bilateral_filtered_image = cv2.bilateralFilter(frame, 7, 150, 150)
    gray_image = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
    return gray_image
	image_sub = cv2.absdiff(preprocessed_image1, preprocessed_image2)
	kernel = numpy.ones((5,5),numpy.uint8)
	close_operated_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
	_, thresholded = cv2.threshold(close_operated_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	_, contours, _ = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(image, contours, -1, (100, 0, 255),2)

	median = cv2.medianBlur(thresholded, 5)
	_, _, angle = cv2.fitEllipse(contour)
	"""
    # Predict the label
    #pred = model.predict(features)

    # Show the label on the screen
    print("The label of the image is: " + str(pred.title()))

    # Pause for 25 ms and keep going until you push q on the keyboard
    if cv2.waitKey(25) == ord('q'):
        break

cap.release() # Release the camera resource
cv2.destroyAllWindows() # Close the image window
