To Run:
1. Open mosaic.m in Matlab. 
2. Make sure to include the rest of the *.m in the same path as the base program, along with having the Computer Vision toolbox installed (necessary for SIFT & matchFeatures).
3. Place all the .jpg images into a folder, with all the file names matching the following pattern: {prefix}_{n}.jpg, with n being positive integers 1...N. 
	NOTE: DO NOT SKIP NUMBERS, IT WILL BREAK THE CODE. DO NOT PUT ANYTHING ELSE IN THE FOLDER EITHER.
4. Run mosaic in Matlab, select the folder that contains all the images, wait for it to process (it might take a while depending on the number/size of images).
5. Afterwards, the resulting mosaic will be displayed and saved in the current folder as {prefix}_mosaic.jpg. 
