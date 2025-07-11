# to reset the models
# to clear the plots
# to clear the log files
# to clear the checkpoints
# to clear _pycache_

import os
import shutil

for root, dirs, files in os.walk("."):
    for d in dirs:
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d))


folder = "outputs/plots"
for f in os.listdir(folder):
    file_path = os.path.join(folder, f)
    if os.path.isfile(file_path):
        os.remove(file_path)

folder = "outputs/gradcam"
for f in os.listdir(folder):
    file_path = os.path.join(folder, f)
    if os.path.isfile(file_path):
        os.remove(file_path)

folder = "outputs/logs"
for f in os.listdir(folder):
    file_path = os.path.join(folder, f)
    if os.path.isfile(file_path):
        os.remove(file_path)

folder = "outputs/checkpoints"
for f in os.listdir(folder):
    file_path = os.path.join(folder, f)
    if os.path.isfile(file_path):
        os.remove(file_path)
        
folder = "webapp/static/uploads"
for f in os.listdir(folder):
    file_path = os.path.join(folder, f)
    if os.path.isfile(file_path):
        os.remove(file_path)
        
folder = "webapp/static/outputs/plots"
for f in os.listdir(folder):
    file_path = os.path.join(folder, f)
    if os.path.isfile(file_path):
        os.remove(file_path)
        
folder = "webapp/static/gradcam"
for f in os.listdir(folder):
    file_path = os.path.join(folder, f)
    if os.path.isfile(file_path):
        os.remove(file_path)

if os.path.exists('outputs/log.txt'):
    os.remove('outputs/log.txt')
    
if os.path.exists('webapp/tempCodeRunnerFile.py'):
    os.remove('webapp/tempCodeRunnerFile.py')
    
if os.path.exists('tempCodeRunnerFile.py'):
    os.remove('tempCodeRunnerFile.py')
    
if os.path.exists("wandb"):
    shutil.rmtree("wandb")