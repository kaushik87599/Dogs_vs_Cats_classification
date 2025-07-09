import torch
from data.dataset import get_dataloaders
from models.custom_cnn import custom_basic_cnn,custom_attention_cnn
from utils.train import training
from utils.evaluate import evaluate_model
from utils.metrics_logger import log_in
from utils.visualization import plot_layers
from utils.evaluate import test_image
from config import device,epochs

model_batch_size = 32
model_num_workers=2
classes={'cats': 0, 'dogs': 1}
classification = {1:'dogs',0:'cats'}


if __name__=="__main__":
    
    log_in('Entered Main Function',write_type='w')
    log_in(f'Device = {device}')
    
    train_dir = 'data/processed/train'
    val_dir = 'data/processed/val'
    
    train_loader, val_loader,test_loader = get_dataloaders(train_dir=train_dir,val_dir=val_dir,batch_size=model_batch_size,num_workers=model_num_workers)
    log_in('Finished Dataloaders')
    basic_model = custom_basic_cnn()
    attention_model = custom_attention_cnn()
    log_in('Finished defining Models')
    
    base_model = basic_model.to(device)
    attention_model = attention_model.to(device)
    
    log_in('Basic Model started training')
    training(base_model,train_loader,val_loader,num_epochs=epochs)
    log_in('Basic Model finished training')
    
    log_in('Attention Model started training')
    training(attention_model,train_loader,val_loader,num_epochs=epochs)
    log_in('Attention Model finished training')
    
    evaluate_model(base_model,test_loader)
    log_in('Basic Model finished evaluating')
    
    evaluate_model(attention_model,test_loader)
    log_in('Attention Model finished evaluating')


    plot_layers(basic_model)
    plot_layers(attention_model)
    
    log_in('Finished testing model')

    
    
    