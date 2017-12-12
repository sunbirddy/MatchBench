In the folder "Dataset":
	1. There are total 8 subfolders.
	2. Each subfolder contains "Images/", "camera.txt", "pairs.txt".
	3. "Images/" is the path of images, and it also contains a file "pose.txt", which saves a matrix with n rows and 12 columns, where n is the number of images, and each row records the camera extrinsic parameter of an image (The pose should include 4x4=16 elements, and we only save the first 3x4=12 elements because the last row of pose is always "0 0 0 1".)
	4. "camera.txt" saves a 3x3 matrix, which is the camera intrinsic parameters.
	5. "pairs.txt" saves a matrix with 14 columns, and each row saves a pair of images that should be matched. The first two columns record the index of two images, and the next 12 columns represent the groundtruth pose transformation between them.