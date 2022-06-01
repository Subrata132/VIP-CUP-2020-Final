

## Get The Data 

To retrain the model:

	1. Put all the images (day time) in 'dataset/train/day/images' folder.
	2. Put all the .txt files (day time) in 'dataset/train/day/labels' folder.
	3. Put all the images (night time) in 'dataset/train/night/images' folder.
	4. Put all the .txt files (night time) in 'dataset/train/night/labels' folder.

## Train

### For First Run only

`python main.py --retrain --model_name='ResNet50' --batch_size=4 --epochs=3 --current_epoch=0`

### For Next runs
Rename `--weights=` argument to last saved model path. model will be saved in `./logs/` directory in the following 
format - `ResNet50_ep_X_Y.h5`, where Y indicates current epoch on previous run.
Largest value of X,Y indicates the latest model.

update `--current_epoch=` each time. (1,2,3... and so on) 

`python main.py --retrain --model_name='ResNet50' --batch_size=4 --epochs=3 --weights='./logs/ResNet50_ep_003_0.h5' --current_epoch=1`


