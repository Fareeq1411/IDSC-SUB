import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

import os
import random
import pandas as pd
from disk_detect import DiscDetector

import cv2
class GlaucomaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["Image Name"]
        label = self.data.iloc[idx]["Label"]
        if str(label.upper()) == "GON+":
            label = 1
        else:
            label = 0

        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        label = float(label)

        if self.transform:
            image = self.transform(image)

        return image, label
    

class resnet50:

    def preprocess_data():
        df = pd.read_csv("Model_2/Labels.csv")

        unique_pt_pos = dict()
        unique_pt_neg = dict()

        extract_patient_data = df[["Image Name", "Patient", "Label"]]

        for img_name, pt,label in zip(extract_patient_data["Image Name"], extract_patient_data["Patient"], extract_patient_data["Label"]):

            if str(label).strip().upper() == "GON+":
                if pt not in unique_pt_pos:
                    unique_pt_pos[pt] = img_name
            else:
                if pt not in unique_pt_neg:
                    unique_pt_neg[pt] = img_name

        pt_id_pos = list(unique_pt_pos.keys())
        pt_id_neg = list(unique_pt_neg.keys())

        random.shuffle(pt_id_pos)
        random.shuffle(pt_id_neg)

        n_pos = len(pt_id_pos)
        n_neg = len(pt_id_neg)

        train_keys = (
            pt_id_pos[:int(0.8 * n_pos)] +
            pt_id_neg[:int(0.8 * n_neg)]
        )

        val_keys = (
            pt_id_pos[int(0.8 * n_pos):int(0.9 * n_pos)] +
            pt_id_neg[int(0.8 * n_neg):int(0.9 * n_neg)]
        )

        test_keys = (
            pt_id_pos[int(0.9 * n_pos):] +
            pt_id_neg[int(0.9 * n_neg):]
        )

        random.shuffle(train_keys)
        random.shuffle(val_keys)
        random.shuffle(test_keys)

        list_img_train = []
        list_img_val = []
        list_img_test = []

        for img_name, pt_id in zip(extract_patient_data["Image Name"], extract_patient_data["Patient"]):
            
            if pt_id in train_keys:
                dataset_dir = "datasets/Images"
                list_img_train.append(os.path.join(dataset_dir, img_name))
            elif pt_id in val_keys:
                dataset_dir = "datasets/Images"
                list_img_val.append(os.path.join(dataset_dir, img_name))
            elif pt_id in test_keys:
                dataset_dir = "datasets/Images"
                list_img_test.append(os.path.join(dataset_dir, img_name))

        output_train = "Model_2/train_data"
        output_val = "Model_2/val_data"
        output_test = "Model_2/test_data"

        random.shuffle(list_img_train)
        random.shuffle(list_img_test)
        random.shuffle(list_img_val)

        DiscDetector.preprocess(list_img_train, output_train)
        DiscDetector.preprocess(list_img_test, output_test)
        DiscDetector.preprocess(list_img_val, output_val)

        output_csv = "Model_2/train.csv"
        series = []
        for img_name in os.listdir("Model_2/train_data"):

            for _, row in df.iterrows():
                if row["Image Name"] == img_name:
                    series.append(row)
            
        train_df = pd.DataFrame(series)
        train_df.to_csv(output_csv, index=False)
        
        output_csv = "Model_2/val.csv"
        series = []

        for img_name in os.listdir("Model_2/val_data"):

            for _, row in df.iterrows():
                if row["Image Name"] == img_name:
                    series.append(row)
            
        train_df = pd.DataFrame(series)
        train_df.to_csv(output_csv, index=False)

        output_csv = "Model_2/test.csv"
        series = []
        for img_name in os.listdir("Model_2/test_data"):

            for _, row in df.iterrows():
                if row["Image Name"] == img_name:
                    series.append(row)
            
        train_df = pd.DataFrame(series)
        train_df.to_csv(output_csv, index=False)
        
    

    def train_model():

        train_csv = "Model_2/train.csv"
        val_csv = "Model_2/val.csv"

        train_data = "Model_2/train_data"
        val_data = "Model_2/val_data"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        transform_gray = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # transform_gray = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     )
        # ])

        train_dataset = GlaucomaDataset(
            csv_file=train_csv,
            img_dir=train_data,
            transform=transform_gray
        )

        val_dataset = GlaucomaDataset(
            csv_file=val_csv,
            img_dir=val_data,
            transform=transform_gray
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        print("Train samples:", len(train_dataset))
        print("Val samples:", len(val_dataset))

        train_df = pd.read_csv(train_csv)
        n_pos = (train_df["Label"] == "GON+").sum()
        n_neg = (train_df["Label"] == "GON-").sum()

        total = n_pos + n_neg
        #NEED TO ADJUST FOR BALANCING
        weight_pos = 1.0
        weight_neg = 1.1

        print("n_pos:", n_pos, "n_neg:", n_neg)
        print("weight_pos:", weight_pos, "weight_neg:", weight_neg)

        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.layer4.parameters():
            param.requires_grad = True

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)

        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss(reduction='none')
        val_criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

        num_epochs = 15

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                optimizer.zero_grad()

                outputs = model(images)

                loss_raw = criterion(outputs, labels)

                sample_weights = torch.where(
                    labels == 0,
                    torch.tensor(weight_neg, device=device),
                    torch.tensor(weight_pos, device=device)
                )

                loss = (loss_raw * sample_weights).mean()

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()

                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.float().unsqueeze(1).to(device)

                    outputs = model(images)
                    loss = val_criterion(outputs, labels)

                    val_loss += loss.item()

                    probs = torch.sigmoid(outputs)
                    preds = (probs >= 0.5).float()

                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total

            print(
                f"Epoch [{epoch+1}/{num_epochs}] | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

        torch.save(model.state_dict(), "resnet50_glaucoma_84.pth")
        print("Model saved.")

    def load_model(model_path):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval() 

        return model, device


    def predict_glaucoma(img_path):

        model_path = "resnet50_glaucoma_84.pth"
        model, device = resnet50.load_model(model_path)

        # transform_gray = transforms.Compose([
        #     transforms.Grayscale(num_output_channels=3),
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     )
        # ])

        transform_gray = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        image = Image.open(img_path).convert("RGB")
        image = transform_gray(image)
        image = image.unsqueeze(0)  
        image = image.to(device)

        with torch.no_grad():
            output = model(image)   
            prob = torch.sigmoid(output).item()

        pred_label = 1 if prob >= 0.55 else 0

        # print("Probability of glaucoma:", prob)
        # print("Predicted label:", pred_label)

        # if pred_label == 1:
        #     print("Prediction: GLAUCOMA")
        # else:
        #     print("Prediction: NORMAL")
        return pred_label, prob
    

    def grayscale_img(img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return gray_3ch


    def test_model():

        image_dir = "Model_2/test_data"
        df = pd.read_csv("Model_2/test.csv")

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for _, row in df.iterrows():
            image_name = row["Image Name"]
            label = row["Label"]
            label = 1 if label == "GON+" else 0

            img_path = os.path.join(image_dir, image_name)
            pred_label, prob = resnet50.predict_glaucoma(img_path)

            if pred_label == label:
                if pred_label == 1:
                    true_positive += 1
                else:
                    true_negative += 1
            else:
                if label == 1:
                    false_negative += 1
                else:
                    false_positive += 1

        total = true_positive + true_negative + false_positive + false_negative

        accuracy = (true_positive + true_negative) / total

        sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

        print(f"TP: {true_positive}, TN: {true_negative}, FP: {false_positive}, FN: {false_negative}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Sensitivity (Recall): {sensitivity * 100:.2f}%")
        print(f"Specificity: {specificity * 100:.2f}%")



    
if __name__ == "__main__":
    # resnet50.preprocess_data()
    # resnet50.train_model()
    resnet50.test_model() 
    