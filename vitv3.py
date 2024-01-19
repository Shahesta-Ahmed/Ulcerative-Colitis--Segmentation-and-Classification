import matplotlib.pyplot as plt
import torch
import torchvision
import os
import json
import numpy as np
from torch import nn
from torchvision import transforms
from helper_functions import set_seeds
from going_modular.going_modular import engine
from torchinfo import summary
from torchvision import datasets
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

#Test 1: Pre-trained ViT with batchsize 32 and learning rate 1e-3 for 30 epochs using a weighted sampler to handle class imbalancing
if __name__ == '__main__':
    # Check if MPS (Apple Silicon GPU) is available, else use CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Selected device: {device}")
    NUM_WORKERS = os.cpu_count()
    BATCH_SIZE = 32

    # Get pretrained weights for ViT-Base and Setup a ViT model instance with pretrained weights
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

    # Change the classifier head
    class_names = ['M0','M1','M2','M3']
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

    # Unfreeze the last N layers of the ViT model
    #N = 2  # Number of layers to unfreeze
    #for i, layer in enumerate(pretrained_vit.encoder.layers[-N:]):
        #for parameter in layer.parameters():
            #parameter.requires_grad = True

    # Initialize the optimizer after setting the trainable layers
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, pretrained_vit.parameters()), lr=1e-3)
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    # Print a summary using torchinfo
    summary(model=pretrained_vit, input_size=(BATCH_SIZE, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"])

    pretrained_vit_transforms = pretrained_vit_weights.transforms()
    print(pretrained_vit_transforms)

    # Load full dataset
    full_dataset = datasets.ImageFolder('/Users/shahestaahmed/Documents/ulcerative_colitis/dataset', transform=pretrained_vit_transforms)

    # Perform stratified split
    train_idx, test_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=full_dataset.targets
    )

    # Create train and test subsets
    train_subset = Subset(full_dataset, train_idx)
    test_subset = Subset(full_dataset, test_idx)

    # Compute class weights
    #class_weights = compute_class_weight('balanced', classes=np.unique(full_dataset.targets), y=full_dataset.targets)
    #class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Create weighted sampler for training data loader
    #train_targets = [full_dataset.targets[i] for i in train_idx]
    #sample_weights = [class_weights[t] for t in train_targets]
    #sampler = WeightedRandomSampler(sample_weights, len(train_subset))
    #train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE,sampler=sampler,num_workers=NUM_WORKERS)

    # Create data loaders
    train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize loss function
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    set_seeds()
    pretrained_vit_results = engine.train(
        model=pretrained_vit,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=60,
        device=device
    )

    final_checkpoint_path = os.path.join('/Users/shahestaahmed/Desktop', 'final_vit_model3_changing_mask_2.pth')
    torch.save(pretrained_vit.state_dict(), final_checkpoint_path)

    # Save training history
    history_path = '/Users/shahestaahmed/Desktop/history_vit_model3_changing_mask_2.json'
    with open(history_path, 'w') as f:
        json.dump(pretrained_vit_results, f)



