# Aerial-Semantic-Segmentation

This is a semantic segmentation problem from https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset. You need to dowload dataset from Kaggle. 
We use Mask R-CNN and U-Net to solve this problem. 
For Mask R-CNN model, you need to upload both original images and labeled mask to your Google Drive, and then use colab for model training and testing.
And then, you need to connect the Colab with Googld Drive to import the dataset into cloud server. For the Mask R-CNN architecture, we use the Pytorch by the https://github.com/pytorch/vision.git with the check point of v0.8.2. We also construct our model referring to Pytorch tutorial https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
