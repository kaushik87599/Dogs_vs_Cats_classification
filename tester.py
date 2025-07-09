from utils.evaluate import test_image
from config import classification,device
import torch
from models.custom_cnn import custom_basic_cnn,custom_attention_cnn
from utils.visualization import Grad_cam
from data.dataset import load_test_image
base_model = custom_basic_cnn()
attention_model=custom_attention_cnn()

load_base_model = torch.load('outputs/checkpoints/best_custom_basic_cnn_model.pth',map_location=device)
load_attention_model = torch.load('outputs/checkpoints/best_custom_attention_cnn_model.pth',map_location=device)

base_model.load_state_dict(load_base_model)
attention_model.load_state_dict(load_attention_model)

base_model.eval()
attention_model.eval()

# test_img = 'examples/Valentine golden retriever.jpeg'
# test_img = 'examples/dog2.jpeg'
# test_img = 'examples/dog3.jpeg'
test_img = 'examples/cats1.jpeg'
# test_img = 'examples/cat2.jpeg'


prediction_base_model = test_image(model=base_model,img = test_img)
prediction_attention_model = test_image(model=attention_model,img = test_img)

image_tensor, orig_image = load_test_image(image_path=test_img)

Grad_cam(model=base_model,image_tensor=image_tensor,orig_image=orig_image)
Grad_cam(model=attention_model,image_tensor=image_tensor,orig_image=orig_image)

print(f'Basic Model Prediction: {classification[prediction_base_model]}')
print(f'Attention Model Prediction: {classification[prediction_attention_model]}')