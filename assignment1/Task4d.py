import torch
import matplotlib.pyplot as plt
import utils
from Task4 import FullyConnectedModel, DeepModel
import dataloaders
import torchvision
from trainer import Trainer


def train_it(model, learning_rate, batch_size, num_epochs, loss_function):
    # Define optimizer (Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate)
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.25,))
    ])
    dataloader_train, dataloader_val = dataloaders.load_dataset(batch_size, image_transform=image_transform)

    trainer = Trainer(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        batch_size=batch_size,
        loss_function=loss_function,
        optimizer=optimizer
    )
    return trainer.train(num_epochs)


# Hyperparameters
learning_rate = .0192
batch_size = 64
num_epochs = 5
loss_function = torch.nn.CrossEntropyLoss()

fc_model = FullyConnectedModel()

deep_model = DeepModel()

_, fc_val_loss_dict = train_it(fc_model, learning_rate, batch_size, num_epochs, loss_function)

_, deep_val_loss_dict = train_it(deep_model, learning_rate, batch_size, num_epochs, loss_function)

# Plot loss
utils.plot_loss(fc_val_loss_dict, label="FC test loss")
utils.plot_loss(deep_val_loss_dict, label="Deep test loss")
plt.ylim([0, 1])
plt.legend()
plt.xlabel("Number of Images Seen")
plt.ylabel("Cross Entropy Loss")
plt.savefig("loss_comp.png")
plt.show()
