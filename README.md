# ASL Real-time System

Broadly the approach used was as follows:
- Pre-processing images: padding, resizing, data augmentation
- Training using CNNs via Transfer Learning - VGG16, ResNet50
    - Feature extraction from a pre-trained network
    - Training Classification Layer
    - Tuning our model
- Experiments, Results and Analysis
    - Loss and Accuracy
- Developing application for executing NN on real time video

## Build instructions
- To run the model with default weights included, just run `livedemo.py`
- change the file names hard-coded in the files to your directory
- change the model names in the files where required to run various models such as vgg16 or resnet50


## Notes
- The dataset used is the Massey University ASL dataset included in the repo in the folder `asl_dataset`
- The raw dataset is split into folders labelled by the alphabet they represent in `label.py`
- The dataset is split into train and test datasets in the folder `split_data` using `create_dataset.py`
- The pre-trained model vgg16 is loaded, top model removed and features extracted in `feature_extraction.py`
- The classification layer is trained and weights saved in `train_classification_layer.py`
- The model is tuned, all layers are frozen except the last convolutional block and the classification layer, the weights learned are loaded and the model re-trained with a SGD optimizer with a very low learning rate to fine-tune the model in `tune_nn.py`
- `live_demo.py` contains the real-time application using open-cv

