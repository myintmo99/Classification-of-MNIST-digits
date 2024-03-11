import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
from torch.utils.tensorboard import SummaryWriter

# Deploying Tensorboard
writer = SummaryWriter("runs/mnist")

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28*28
hidden_size = 100
num_classes = 10
epochs = 2
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root='D:/Data D/Online Learning/Pytorch', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='D:/Data D/Online Learning/Pytorch', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

eg = iter(train_loader)
sample, label = next(eg)
print(sample.shape, label.shape)
print(sample[1].shape)
print(sample[1][0].shape)

# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(sample[i][0], cmap='gray')
# plt.show()

# Need to use 'make_grid' to display batches of images
img_grid = torchvision.utils.make_grid(sample)
writer.add_image('Mnist Images', img_grid)

# Model Construction
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Adding Graph to Tensorboard
writer.add_graph(model, sample.reshape(-1,28*28))
# writer.close()
# sys.exit()

# Training the model
running_loss = 0.0
running_correct = 0
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        pred = model(images)
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,predictions = torch.max(pred, 1)
        running_correct += (predictions==labels).sum().item()

        if (i+1)%100 == 0:
            print(f'epoch {epoch+1}/{epochs}: step{i+1}/{len(train_loader)}, loss = {loss.item():.4f}')
            writer.add_scalar('Training Loss', running_loss/100, epoch*len(train_loader)+i)
            writer.add_scalar('Accuracy', running_correct/100, epoch*len(train_loader)+i)
            running_loss=0.0
            running_correct=0

image_labels = []
preds = []

# Testing Model
with torch.no_grad():
    correct = 0
    samples = 0
    for images,labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _,predictions = torch.max(outputs, 1)
        samples += images.shape[0]
        correct += predictions.eq(labels).sum().item()

        predicted_classes = [f.softmax(output, dim=0) for output in outputs]
        preds.append(predicted_classes)
        image_labels.append(predictions)

    preds = torch.cat([torch.stack(batch) for batch in preds])
    image_labels = torch.cat(image_labels)
    acc = 100*correct / samples
    print(f'Accuracy : {acc}')

    for i in range(10):
        labels_i = image_labels==i
        preds_i = preds[:,i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()