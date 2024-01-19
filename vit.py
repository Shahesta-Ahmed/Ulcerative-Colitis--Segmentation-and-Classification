
import matplotlib.pyplot as plt
import torch
import torchvision
import os
import requests
import json
from torch import nn
from torchvision import transforms
from helper_functions import set_seeds
from going_modular.going_modular import engine
from going_modular.going_modular.predictions import pred_and_plot_image
from torchinfo import summary
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from helper_functions import plot_loss_curves
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler, Subset


# Check if MPS (Apple Silicon GPU) is available, else use CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Selected device: {device}")

# Get pretrained weights for ViT-Base and Setup a ViT model instance with pretrained weights
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# Change the classifier head
class_names = ['M0','M1','M2','M3']
pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

# Unfreeze the last N layers of the ViT model
N = 2  # Number of layers to unfreeze
num_layers = len(pretrained_vit.encoder.layers)  # Total number of layers in the encoder
for i, layer in enumerate(pretrained_vit.encoder.layers):
    if i >= num_layers - N:
        for parameter in layer.parameters():
            parameter.requires_grad = True

# Initialize the optimizer after setting the trainable layers
optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, pretrained_vit.parameters()), lr=1e-4)

# Print a summary using torchinfo
summary(model=pretrained_vit, input_size=(32, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"])

# Setup directory paths to train and test images
train_dir = '/Users/shahestaahmed/Documents/ulcerative_colitis/uc/train'
test_dir = '/Users/shahestaahmed/Documents/ulcerative_colitis/uc/test'

# Get automatic transforms from pretrained ViT weights
pretrained_vit_transforms = pretrained_vit_weights.transforms()
print(pretrained_vit_transforms)

if __name__ == '__main__':
    NUM_WORKERS = os.cpu_count()
    BATCH_SIZE = 64

    # Load full dataset
    full_dataset = datasets.ImageFolder('/Users/shahestaahmed/Documents/ulcerative_colitis/uc', transform=pretrained_vit_transforms)

    # Perform stratified split
    train_idx, test_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.4,
        random_state=42,
        stratify=full_dataset.targets
    )

    # Create train and test subsets
    train_subset = Subset(full_dataset, train_idx)
    test_subset = Subset(full_dataset, test_idx)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(full_dataset.targets), y=full_dataset.targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Create weighted sampler for training data loader
    train_targets = [full_dataset.targets[i] for i in train_idx]
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(train_subset))

    # Create data loaders
    train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, pretrained_vit.parameters()), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    set_seeds()
    pretrained_vit_results = engine.train(
        model=pretrained_vit,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=10,
        device=device
    )

    checkpoint_dir = '/Users/shahestaahmed/Documents/checkpoints/learning rate'
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(10):
        checkpoint_path = os.path.join(checkpoint_dir, f'vitv3_{epoch}.pth')
        torch.save(pretrained_vit.state_dict(), checkpoint_path)

    # Save training history as a JSON file
    history = {
    'train_loss': pretrained_vit_results['train_loss'],
    'test_loss': pretrained_vit_results['test_loss'],
    'train_acc': pretrained_vit_results['train_acc'],
    'test_acc': pretrained_vit_results['test_acc'],
    }

# Save training history as a JSON file
    history_path = '/Users/shahestaahmed/Documents/history/history_vitv3_mask.json'
    with open(history_path, 'w') as f:
        json.dump(history, f)
