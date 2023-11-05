import torch
import torchvision
import torchmetrics
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, mobilenet_v3_small
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from pytorch_lightning.callbacks import ModelCheckpoint

class Fashion_MNIST_ResNet(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = resnet50(weights=None, num_classes=10)
    self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    self.criterion = nn.CrossEntropyLoss()
    self.acc = MulticlassAccuracy(num_classes=10)
    self.f1 = MulticlassF1Score(num_classes=10)

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self(inputs)
    loss = self.criterion(outputs, targets)
    self.log("train_loss", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self(inputs)
    loss = self.criterion(outputs, targets)
    self.acc.update(outputs, targets)
    self.f1.update(outputs, targets)
    self.log("val_loss", loss)
    return loss

  def on_validation_epoch_end(self):
    e_loss = self.trainer.callback_metrics.get('val_loss')
    t_loss = self.trainer.callback_metrics.get('train_loss')
    e_acc = self.acc.compute()
    e_f1 = self.f1.compute()
    self.log("val_acc", e_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log("val_f1", e_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    print(f"\n\nEpoch: {self.current_epoch} - Metrics: ")
    print(f"Training loss: {t_loss}, Validation loss:{e_loss:.4f}, Validation accuracy: {e_acc:.4f}, Validation F1: {e_f1:.4f}\n")
    self.acc.reset()
    self.f1.reset()

  def configure_optimizers(self):
    return optim.AdamW(self.model.parameters(), lr=1e-3)
  


class Fashion_MNIST_MobileNet(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = mobilenet_v3_small(weights=None, num_classes=10)
    self.model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    self.criterion = nn.CrossEntropyLoss()
    self.acc = MulticlassAccuracy(num_classes=10)
    self.f1 = MulticlassF1Score(num_classes=10)

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self(inputs)
    loss = self.criterion(outputs, targets)
    self.log("train_loss", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self(inputs)
    loss = self.criterion(outputs, targets)
    self.acc.update(outputs, targets)
    self.f1.update(outputs, targets)
    self.log("val_loss", loss)
    return loss

  def on_validation_epoch_end(self):
    e_loss = self.trainer.callback_metrics.get('val_loss')
    t_loss = self.trainer.callback_metrics.get('train_loss')
    e_acc = self.acc.compute()
    e_f1 = self.f1.compute()
    self.log("val_acc", e_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log("val_f1", e_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    print(f"\n\nEpoch: {self.current_epoch} - Metrics: ")
    print(f"Training loss: {t_loss}, Validation loss:{e_loss:.4f}, Validation accuracy: {e_acc:.4f}, Validation F1: {e_f1:.4f}\n")
    self.acc.reset()
    self.f1.reset()

  def configure_optimizers(self):
    return optim.AdamW(self.model.parameters(), lr=1e-3)