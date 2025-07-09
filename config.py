import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 25

classes={'cats': 0, 'dogs': 1}
classification = {1:'dogs',0:'cats'}
