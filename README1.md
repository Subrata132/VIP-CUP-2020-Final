# BUET_Endgame Folder Description
The folder should contain the followings files and folders.
	Files:
		1. BUET_Endgame.pdf
		2. detection.py
		3. readme.md
		4. requirement.txt
		5. run.py
		6. tracking.py
	Folders:
		1. callbacks
		2. data
		3. dataset
		4. logs
		5. loss
		6. model
		7. results
		8. saved_model
		9. stuffs
		
# Requirement
You're requested to see the 'requirement.txt' file to know if you're up to date. To ensure you're up to date, run:
		'pip install -r requirement.txt'
		
# Get The Data 

To retrain the model:

	1. Put all the images (day time) in 'BUET_Endgame/dataset/train/day/images' folder.
	2. Put all the .txt files (day time) in 'BUET_Endgame/dataset/train/day/labels' folder.
	3. Put all the images (night time) in 'BUET_Endgame/dataset/train/night/images' folder.
	4. Put all the .txt files (night time) in 'BUET_Endgame/dataset/train/night/labels' folder.

To test the model:
	
	1. Put all the images (day/night time) in 'BUET_Endgame/dataset/test/images' folder.
	
# Train The Network

After the successful completation of previous two sections you may train the network once again. You can 'SKIP' this portion as it's already trained.  

To retrain the network do the followings:
			1. Open command window in 'BUET_Endgame' folder.
			2. Type 'python main.py --retrain --batch_size=8 --epochs=120 --weights='./logs/-----' --current_epoch=0 --multigpu' and hit 'Enter'.
						
					* You may change the batch_size suitable for the machine.
					* exclude --multigpu if multiple GPU is not available.
					* weights can be loaded in case of discontinous trainning. For this replace the '----'  portion by most recently saved .h5 file(can be found in log folder)
					* change the current_epoch accordingly.

# Test The Network



We are expecting that the image names should be in the following format,
		<camera name>+<image serial number>+<.extention> e.g: 'konya_azizmahmut_1.png' or 'CLIP_20200628-210253_000000.jpg'
		
	If the image serials are padded with zeros (like training images) the code will need slight modifiaction. In that case we request you to contact us.

To test the model:
			1. Open command window in BUET_Endgame folder.
			2. Type 'python detection.py --extension=.png'  (--extention is set to .jpg as default)
			

# Tracking

After running the 'detection.py' successfully you can run the following command in 'BUET_Endgame' folder to see the output of tracking.
			'python tracking.py'
			
	* For this part we expect the set of images to be of reasonable fps (around 10 fps, e.g primary test and test sets)
	* We are expecting a sequence of images from a single camera at a time.
	
# Result 

After the successful completation of 'Test The Network' part check the 'BUET_Endgame/results/labels' folder. 
This folder should consist some of .txt file (equal of the number of images in 'dataset/test' folder, one for each image )

A .mp4 file  named 'video_output.mp4' in 'BUET_Endgame/results/video' folder should be found.
A .mp4 file  named 'tracked.mp4' in 'BUET_Endgame/results/video' folder should be found.


# For Any Query
You're welcome to contact us:
	1. subrata.biswas@ieee.org
	2. shouborno@ieee.org
	3. Shakhrulsiam268@gmail.com
	4. omartawhid97@gmail.com 
