import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# === 1. VERİ SETİNİ YÜKLEME VE ÖN İŞLEME ===
img_size = 150  # Tüm görüntüleri bu boyuta yeniden boyutlandırıyoruz
batch_size = 32  # Model eğitimi sırasında kaç görüntünün birlikte işleneceğini belirler

# Görüntüleri uygun şekilde ölçeklendirme ve artırma işlemi (Data Augmentation)
datagen = ImageDataGenerator(
    rescale=1./255,  # Piksel değerlerini 0-1 aralığına normalleştirir
    rotation_range=20, # Rastgele döndürme
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Veriyi %80 eğitim, %20 doğrulama olarak ayır
)

# Eğitim verileri
train_data = datagen.flow_from_directory(
    "dataset/train",  # Görüntülerin bulunduğu klasör
    target_size=(img_size, img_size),  # Yeniden boyutlandırma
    batch_size=batch_size,
    class_mode="categorical",  # Çok sınıflı çıktı
    subset="training"
)

# Doğrulama verileri
val_data = datagen.flow_from_directory(
    "dataset/train",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# === 2. CNN MODELİNİ OLUŞTURMA ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Aşırı öğrenmeyi önlemek için Dropout
    Dense(4, activation='softmax')  # 4 farklı sınıf olduğu için 4 çıktı kullanıyoruz
])

# Modeli derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === 3. MODELİ EĞİTME ===
history = model.fit(train_data, validation_data=val_data, epochs=20)

# === 4. MODELİN BAŞARIMINI GÖRSELLEŞTİRME ===
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

# === 5. MODELİ TEST ETME ===
from tensorflow.keras.preprocessing import image

# Test edilecek görüntü dosya yolu
img_path = "dataset/test/no_tumor/image1.jpg"

# Görüntüyü yükle ve modele uygun hale getir
img = image.load_img(img_path, target_size=(img_size, img_size))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizasyon

# Model ile tahmin yap
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)
labels = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
print("Tahmin edilen sınıf:", labels[predicted_class[0]])

# === 6. MODELİ KAYDETME ===
model.save("brain_tumor_model.h5")
print("Model başarıyla kaydedildi!")
