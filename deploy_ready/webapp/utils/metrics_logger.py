from time import time
import pandas as pd
from datetime import datetime
class Logger():
    ''' module for logging metrics 
    used by the trainer module '''
    def __init__(self, model_name):
        self.model_name = model_name
        self.start_time = time()

    
    def log_metrics(self,epoch, train_loss, train_accuracy, val_loss, val_accuracy):
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        time_elapsed = time()-self.start_time
        
        data = {'Epoch': [epoch+1],'Time': [time_elapsed], 'Train Loss': [train_loss], 'Train Accuracy': [train_accuracy], 'Val Loss': [val_loss], 'Val Accuracy': [val_accuracy]}
        
        data_pt = pd.DataFrame(data,columns=['Epoch', 'Time', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy'])
        
        data_pt.to_csv(f'outputs/logs/{self.model_name}.csv', mode='a', index=False,header=False)
        
    def log_best_metrics(self,epoch, train_loss, train_accuracy, val_loss, val_accuracy):
        # print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        time_elapsed = time()-self.start_time
        data = {'Epoch': [epoch+1],'Time': [time_elapsed], 'Train Loss': [train_loss], 'Train Accuracy': [train_accuracy], 'Val Loss': [val_loss], 'Val Accuracy': [val_accuracy]}
        data_pt = pd.DataFrame(data)
        data_pt.to_csv(f'outputs/logs/{self.model_name}_best.csv', mode='w', index=False,header=False)
        
def log_in(message:str,write_type='a'):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('outputs/log.txt', write_type) as log_file:
        log_file.write(f'Time : {timestamp} || {message}\n')
    
