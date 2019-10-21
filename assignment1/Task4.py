import torch
import matplotlib.pyplot as plt
import utils
import dataloaders
import torchvision
from trainer import Trainer
import numpy as np

torch.random.manual_seed(0)


class FullyConnectedModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # We are using 28x28 greyscale images.
        num_input_nodes = 28*28
        # Number of classes in the MNIST dataset
        num_classes = 10

        # Define our model
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_input_nodes, num_classes),
        )

    def forward(self, x):
        # Runs a forward pass on the images
        x = x.view(-1, 28*28)
        out = self.classifier(x)
        return out


class DeepModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # We are using 28x28 greyscale images.
        num_input_nodes = 28*28
        # Number of classes in the MNIST dataset
        num_classes = 10
        # Number of nodes in hidden layer
        num_hidden_nodes = 64

        # Define our model
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_input_nodes, num_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_nodes, num_classes),
        )

    def forward(self, x):
        # Runs a forward pass on the images
        x = x.view(-1, 28*28)
        out = self.classifier(x)
        return out


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    learning_rate = .0192
    num_epochs = 5

    # Use CrossEntropyLoss for multi-class classification
    loss_function = torch.nn.CrossEntropyLoss()

    # Model definition
    model = FullyConnectedModel()

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
    train_loss_dict, val_loss_dict = trainer.train(num_epochs)

    # Plot loss
    utils.plot_loss(train_loss_dict, label="Train Loss")
    utils.plot_loss(val_loss_dict, label="Test Loss")
    plt.ylim([0, 1])
    plt.legend()
    plt.xlabel("Number of Images Seen")
    plt.ylabel("Cross Entropy Loss")
    plt.savefig("training_loss.png")
    plt.show()

    # Plot weights
    weight = next(model.classifier.children()).weight.data

    for i in range(10):
        mat = np.reshape(weight[i], (28, 28))
        plt.imshow(mat, cmap="gray")
        plt.savefig("weights_{}.png".format(i))
    torch.save(model.state_dict(), "deep_model.torch")
    final_loss, final_acc = utils.compute_loss_and_accuracy(
        dataloader_val, model, loss_function)
    print("Final Test Cross Entropy Loss: {}. Final Test accuracy: {}".format(final_loss, final_acc))
