import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import GestureDetectDataset
import matplotlib.pyplot as plt

class CustomYoloLoss(nn.Module):
    def __init__(self):
        super(CustomYoloLoss, self).__init__()
        self.bbox_loss = nn.SmoothL1Loss()
        self.obj_loss = nn.BCEWithLogitsLoss()
        self.class_loss = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = CustomYoloLoss()
    
    def forward(self, predictions, targets):
        pred_bboxes, pred_obj, pred_class = predictions
        true_bboxes, true_obj, true_class = targets

        bbox_loss = self.bbox_loss(pred_bboxes, true_bboxes)
        obj_loss = self.obj_loss(pred_obj, true_obj)
        class_loss = self.class_loss(pred_class, true_class)

        total_loss = bbox_loss + obj_loss + class_loss
        return total_loss

class Model():
    def __init__(self, model_path, dataset_path):
        self.dataset_path = dataset_path
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.get_dataloaders()
        self.model = torch.load(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def get_dataloaders(self):
        train_dataset = GestureDetectDataset(self.dataset_path, mode='train')
        val_dataset = GestureDetectDataset(self.dataset_path, mode='val')
        test_dataset = GestureDetectDataset(self.dataset_path, mode='test')

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader
    def train(self, num_epochs = 10, plot = True):

        self.model.train()

        self.model = self.model.to(self.device)

        train_loss_history = []
        val_loss_history = []

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            loop = tqdm(self.train_dataloader, total=len(self.train_dataloader), leave=False)

            for images, bboxes, class_labels in loop:
                images = images.to(self.device)
                bboxes = torch.tensor(bboxes).to(self.device)
                class_labels = torch.tensor(class_labels).to(self.device)

                self.optimizer.zero_grad()

                predictions = self.model(images)

                targets = (bboxes, torch.ones_like(bboxes[..., :1]), class_labels)

                loss = self.criterion(predictions, targets)
                epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
                loop.set_postfix(loss=loss.item())

            train_loss_history.append(epoch_loss / len(self.train_dataloader))
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss / len(self.train_dataloader)}")

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, bboxes, class_labels in self.val_dataloader:
                    images = images.to(self.device)
                    bboxes = torch.tensor(bboxes).to(self.device)
                    class_labels = torch.tensor(class_labels).to(self.device)

                    predictions = self.model(images)

                    targets = (bboxes, torch.ones_like(bboxes[..., :1]), class_labels)
                    loss = self.criterion(predictions, targets)
                    val_loss += loss.item()

            val_loss_history.append(val_loss / len(self.val_dataloader))
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss / len(self.val_dataloader)}")
        
        if plot:
            self.show_plot(num_epochs, train_loss_history, val_loss_history)
            
        return train_loss_history, val_loss_history

    def show_plot(self, num_epochs, train_loss_history, val_loss_history):
        
        epochs = range(1, num_epochs + 1)
        plt.plot(epochs, train_loss_history, label="Training Loss", color='blue')
        plt.plot(epochs, val_loss_history, label="Validation Loss", color='red')   
        
        plt.title('Training and Validation Losses Over Epochs')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss') 
        
        plt.legend()   
        plt.show()
    def evaluate(self, save = True):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, bboxes, class_labels in self.test_dataloader:
                images = images.to(self.device)
                bboxes = torch.tensor(bboxes).to(self.device)
                class_labels = torch.tensor(class_labels).to(self.device)

                predictions = self.model(images)

                targets = (bboxes, torch.ones_like(bboxes[..., :1]), class_labels)
                loss = self.criterion(predictions, targets)
                test_loss += loss.item()

        print(f"Test Loss: {test_loss / len(self.test_dataloader)}")

        if save:
            torch.save(self.model.state_dict(), "fine_tuned_yolo.pt")