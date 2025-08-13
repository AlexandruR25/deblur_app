# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 09:25:37 2025

@author: alexandru.raducu02
"""

"""
Proiect Licență: Înlăturarea defocalizării din imagini cu rețele neuronale profunde
Autor: [Raducu Alexandru]
Coordonator: [Nume Profesor]
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tkinter as tk
from tkinter import messagebox
import random
from PIL import Image, ImageDraw, ImageFilter
from tkinter import ttk
import time
import requests
from io import BytesIO
import re

# Configurație implicită
DEFAULT_CONFIG = {
    "train_dir": "data/train",
    "val_dir": "data/val",
    "model_save_dir": "models",
    "results_dir": "results",
    "default_epochs": 100,
    "default_batch_size": 8,
    "default_model_path": "models/best_model.pth",
    "save_every_epoch": False
}

IMAGE_SIZE = (256, 256)  # Dimensiunea imaginii 255x255
MAX_RETRIES = 3  # Maximul de încercări în cazul unui eșec
REQUEST_TIMEOUT = 5

train_sharp_dir = "data/train/sharp"
train_blur_dir = "data/train/blur"
test_blur_dir = "data/test/blur"
results_deblur_dir = "results"

class DefocusDataset(Dataset):
    """Set de date pentru antrenare cu detectare automată a structurii"""
    def __init__(self, blur_dir, sharp_dir):
        self.blur_paths = sorted([os.path.join(blur_dir, f) for f in os.listdir(blur_dir)])
        self.sharp_paths = sorted([os.path.join(sharp_dir, f) for f in os.listdir(sharp_dir)])

    def __len__(self):
        return min(len(self.blur_paths), len(self.sharp_paths))

    def __getitem__(self, idx):
        blur = cv2.imread(self.blur_paths[idx]).astype(np.float32) / 255.0
        sharp = cv2.imread(self.sharp_paths[idx]).astype(np.float32) / 255.0
        return (
            torch.from_numpy(blur).permute(2,0,1),
            torch.from_numpy(sharp).permute(2,0,1)
        )

class AutoDeblurUNet(nn.Module):
    """Arhitectură U-Net avansată cu blocuri reziduale"""
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        
        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder
        u1 = self.up1(b)
        c1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(c1)
        
        u2 = self.up2(d1)
        c2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(c2)
        
        return self.final(d2)

def automated_trainer():
    """Antrenare automată cu reluare și salvare progres"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispozitiv utilizat: {device}")

    # Verificare structură directoare
    os.makedirs(DEFAULT_CONFIG["model_save_dir"], exist_ok=True)

    # Inițializare dataset și loader
    train_dataset = DefocusDataset(
        os.path.join(DEFAULT_CONFIG["train_dir"], "blur"),
        os.path.join(DEFAULT_CONFIG["train_dir"], "sharp")
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=DEFAULT_CONFIG["default_batch_size"],
        shuffle=True
    )

    # Inițializare model, optimizator, loss
    model = AutoDeblurUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()
    best_loss = float('inf')

    # Fișiere salvate
    model_path = DEFAULT_CONFIG["default_model_path"]
    optimizer_path = os.path.join(DEFAULT_CONFIG["model_save_dir"], "optimizer.pth")
    epoch_file_path = os.path.join(DEFAULT_CONFIG["model_save_dir"], "epoch.txt")

    # Verificare existență model/optimizator și epocă
    start_epoch = 0
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model încărcat din {model_path}")

        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
            print("Starea optimizatorului încărcată din models\\optimizer.pth")

        if os.path.exists(epoch_file_path):
            with open(epoch_file_path, "r") as f:
                start_epoch = int(f.read().strip())
            print(f"Model încărcat din ultima epocă salvată: {start_epoch}")

    # Setare capăt epoci noi
    total_epochs = start_epoch + 1 + DEFAULT_CONFIG["default_epochs"]

    # Buclă de antrenament
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0.0
        images_processed = 0

        for batch_idx, (blur, sharp) in enumerate(train_loader):
            blur, sharp = blur.to(device), sharp.to(device)

            optimizer.zero_grad()
            outputs = model(blur)
            loss = criterion(outputs, sharp)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            images_processed += blur.size(0)

            if batch_idx % 100 == 0:
                print(f"Epoca {epoch}/{total_epochs - 1} - Batch {batch_idx}/{len(train_loader)} - Imagini procesate: {images_processed}")

        avg_loss = total_loss / len(train_loader)

        # Salvare model optim
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f"Model salvat la: {model_path}")

        # Salvare stare optimizator și epocă
        torch.save(optimizer.state_dict(), optimizer_path)
        with open(epoch_file_path, "w") as f:
            f.write(str(epoch))

        print(f"Epoca {epoch}/{total_epochs - 1} | Loss: {avg_loss:.10f}")
        print(f"Imagini procesate în epoca {epoch}: {images_processed}/{len(train_loader.dataset)}")

    print("Antrenamentul a fost finalizat!")

def automated_processor():
    """Procesare automată a imaginilor de test"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoDeblurUNet().to(device)
    
    try:
        model.load_state_dict(torch.load(DEFAULT_CONFIG["default_model_path"], map_location=device))
    except FileNotFoundError:
        print("Eroare: Model preantrenat nu există! Rulați mai întâi antrenarea (opțiunea 1).")
        return
    
    # Procesare imagini test
    test_dir = "data/test/blur"
    os.makedirs(DEFAULT_CONFIG["results_dir"], exist_ok=True)
    
    for img_file in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_file)
        img = cv2.imread(img_path).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
        
        output_img = output.squeeze().permute(1,2,0).cpu().numpy() * 255
        cv2.imwrite(os.path.join(DEFAULT_CONFIG["results_dir"], f"deblurred_{img_file}"), output_img)
 
def afiseaza_comparatie_imagini():
    delay=3000
    target_height=512
    window_x = 100
    window_y = 100
    blurred_folder = "data/test/blur"
    deblurred_folder = "results"
    
    blurred_images = sorted([f for f in os.listdir(blurred_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    deblurred_images = sorted([f for f in os.listdir(deblurred_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    total_images = min(len(blurred_images), len(deblurred_images))

    if len(blurred_images) != len(deblurred_images):
        print("⚠️ Număr diferit de imagini în foldere! Continuăm cu perechi potrivite.")

    for i, (blur_name, deblur_name) in enumerate(zip(blurred_images, deblurred_images), start=1):
        blur_path = os.path.join(blurred_folder, blur_name)
        deblur_path = os.path.join(deblurred_folder, deblur_name)

        blur_img = cv2.imread(blur_path)
        deblur_img = cv2.imread(deblur_path)

        if blur_img is None or deblur_img is None:
            print(f"❌ Nu s-a putut citi una dintre imaginile: {blur_name}, {deblur_name}")
            continue

        def resize_to_height(img, height):
            h, w = img.shape[:2]
            scale = height / h
            new_w = int(w * scale)
            return cv2.resize(img, (new_w, height))

        blur_resized = resize_to_height(blur_img, target_height)
        deblur_resized = resize_to_height(deblur_img, target_height)

        # Concatenăm imaginile
        comparatie = np.hstack((blur_resized, deblur_resized))

        # Adăugăm spațiu sus pentru titluri (ex: 50px înălțime)
        padding = 50
        h, w = comparatie.shape[:2]
        comparatie_padded = np.zeros((h + padding, w, 3), dtype=np.uint8)
        comparatie_padded[padding:, :, :] = comparatie

        # Adăugăm etichete deasupra fiecărei jumătăți
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 255, 255)

        mid_x = blur_resized.shape[1]
        cv2.putText(comparatie_padded, "Blurred Image", (20, 35), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(comparatie_padded, "Processed Image", (mid_x + 20, 35), font, font_scale, color, thickness, cv2.LINE_AA)

        # Adăugăm indicator de progres în titlul ferestrei
        window_name = f"[{i}/{total_images}] Comparație: {blur_name} vs {deblur_name}"
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, w, h + padding)
        cv2.moveWindow(window_name, window_x, window_y)
        cv2.imshow(window_name, comparatie_padded)

        print(f"Afișez: {blur_name}  <->  {deblur_name}  [{i}/{total_images}]")

        # Așteptăm fie 3 secunde, fie o apăsare de tastă
        key = cv2.waitKey(delay)
        if key == 27:  # Dacă utilizatorul apasă ESC, ieșim
            break

        cv2.destroyWindow(window_name)

    cv2.destroyAllWindows()
    
def get_max_image_number(directory):
    """Funcție pentru a obține cel mai mare număr de imagine existent"""
    existing_files = os.listdir(directory)
    max_num = 0
    for file in existing_files:
        match = re.search(r"(\d+)\.png$", file)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)
    return max_num

def handle_images(num_images, progress_var, root):
    # Asigurăm că directoarele există
    os.makedirs(train_sharp_dir, exist_ok=True)
    os.makedirs(train_blur_dir, exist_ok=True)
    os.makedirs(test_blur_dir, exist_ok=True)

    # Obținem cel mai mare număr pentru a evita suprascrierea
    max_sharp_num = get_max_image_number(train_sharp_dir)
    max_blur_num = get_max_image_number(train_blur_dir)
    max_test_blur_num = get_max_image_number(test_blur_dir)

    # Pornim numerotarea de la cel mai mare număr existent + 1
    start_num = max(max_sharp_num, max_blur_num, max_test_blur_num) + 1

    failed_downloads = 0  # numărăm erorile de descărcare

    for i in range(start_num, start_num + num_images):
        try:
            # 1. Descarcă o imagine aleatorie de pe picsum.photos
            url = f"https://picsum.photos/600/400?random={i}"
            img = None
            for attempt in range(MAX_RETRIES):
                try:
                    response = requests.get(url, timeout=REQUEST_TIMEOUT)
                    response.raise_for_status()
                    if 'image' not in response.headers['Content-Type']:
                        raise ValueError("Răspunsul nu este o imagine validă")
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    break
                except Exception as e:
                    print(f"Încercare {attempt + 1} pentru imaginea {i} eșuată: {str(e)}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(1)
                    if attempt == MAX_RETRIES - 1:
                        failed_downloads += 1
                        print(f"❌ Eșuat la descărcarea imaginii {i}, va fi ignorată.")
                        if failed_downloads >= 3:  # Dacă avem prea multe erori de descărcare, oprim procesul
                            print("Prea multe eșecuri la descărcarea imaginilor, procesul va fi oprit.")
                            return

            if img is None:
                print(f"Nu s-a putut descărca imaginea {i}, trecem la următoarea.")
                continue

            # 2. Redimensionare imagine
            img = img.resize(IMAGE_SIZE)

            # Verifică dacă imaginea are dimensiunile corecte
            if img.size != IMAGE_SIZE:
                print(f"Imaginea {i} nu are dimensiunile corecte după redimensionare, ajustăm...")
                img = img.resize(IMAGE_SIZE)

            # 3. Salvăm imaginea sharp (originală)
            sharp_path = os.path.join(train_sharp_dir, f"sharp_image_{i}.png")
            img.save(sharp_path)

            # 4. Aplicăm blur și salvăm imaginea blurată
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=1))
            blur_train_path = os.path.join(train_blur_dir, f"blurred_image_{i}.png")
            blurred_img.save(blur_train_path)
            
            # 5. Salvăm și în directorul de test
            blur_test_path = os.path.join(test_blur_dir, f"blurred_image_{i}.png")
            blurred_img.save(blur_test_path)

            # 6. Actualizăm bara de progres
            progress_var.set((i - start_num + 1) / num_images * 100)
            root.update()  # Actualizăm interfața

            # Simulăm un delay pentru a vizualiza progresul
            time.sleep(0.01)

        except Exception as e:
            print(f"Eroare procesare imagine {i}: {str(e)}")

    print(f"{num_images} imagini au fost adăugate și procesate.")
    
def afiseaza_meniu_text():
    
    def delete_imagini(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    while True:
        print("\n=== Sistem Autonom Deblurring Defocus ===")
        print("1. Antrenare model")
        print("2. Procesare imagini test")
        print("3. Adăugare imagini")
        print("4. Ștergere poze")
        print("5. Ieșire")
        
        optiune = input("Alege o opțiune (1-5): ").strip()

        if optiune == '1':
            print("\nInițializare antrenare cu parametri default...")
            automated_trainer()

        elif optiune == '2':
            print("\nProcesare automată a imaginilor din data/test/blur...")
            automated_processor()
            afiseaza_comparatie_imagini()

        elif optiune == '3':
            try:
                num_images = int(input("Introduceți numărul de imagini de adăugat: "))
                print(f"\nAdăugare {num_images} imagini...")
                handle_images(num_images)
                print(f"{num_images} imagini au fost adăugate și procesate.")
            except ValueError:
                print("Eroare: Introdu un număr valid!")

        elif optiune == '4':
            print("Ștergere imagini...")
            delete_imagini(test_blur_dir)
            delete_imagini(train_blur_dir)
            delete_imagini(train_sharp_dir)
            delete_imagini(results_deblur_dir)
            print("Imaginile au fost șterse cu succes.")

        elif optiune == '5':
            print("La revedere!")
            break

        else:
            print("Opțiune invalidă. Încearcă din nou.")


# Apelează funcția pentru a porni aplicația
if __name__ == "__main__":
    afiseaza_meniu_text()

