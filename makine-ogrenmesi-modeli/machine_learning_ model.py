import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt
import time
import traceback
from itertools import product  # parametre kombinasyonları için

# oneDNN optimizasyonlarını devre dışı bırak
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# dosya uzantıları
VIDEO_DIR = "C:/Users/shilo/OneDrive/Masaüstü/modelbir/videos"
LABEL_FILE = "C:/Users/shilo/OneDrive/Masaüstü/modelbir/labels.csv"
MODEL_SAVE_DIR = "C:/Users/shilo/OneDrive/Masaüstü/modelbir"

FRAME_SIZE = (64, 64)
MAX_FRAMES = 20
EXPECTED_VIDEO_EXTENSION = ".mp4"
MIN_SAMPLES_PER_CLASS_FOR_STRATIFY = 2

# fonksiyonlar
def extract_frames(video_path, max_frames=MAX_FRAMES, frame_size=FRAME_SIZE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"UYARI: Video açılamadı! -> {video_path}")
        frames = []
    else:
        frames = []
        count = 0
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                frame_resized = cv2.resize(frame, frame_size)
                frame_normalized = frame_resized / 255.0
                frames.append(frame_normalized)
                count += 1
            except Exception as e:
                print(f"HATA: Kare işlenirken sorun ({video_path}): {e}")
                break
        cap.release()

    while len(frames) < max_frames:
        # Eksik kareleri siyah karelerle doldur
        frames.append(np.zeros((frame_size[0], frame_size[1], 3)))
    return np.array(frames)

def load_dataset():
    if not os.path.exists(LABEL_FILE):
        raise FileNotFoundError(f"Etiket dosyası bulunamadı: {LABEL_FILE}")
    if not os.path.isdir(VIDEO_DIR):
        raise FileNotFoundError(f"Video dizini bulunamadı veya bir dizin değil: {VIDEO_DIR}")
    print(f"Etiket dosyası okunuyor: {LABEL_FILE}")
    print(f"Video dizini kontrol ediliyor: {VIDEO_DIR}")

    try:
        df = pd.read_csv(LABEL_FILE, encoding='cp1254', delimiter=';')
        print("CSV dosyası 'cp1254', ';' ile okundu.")
    except Exception as e:
        print(f"HATA: CSV dosyası okunamadı. Hata: {e}")
        raise

    video_paths_initial = []
    string_labels_initial = []
    print(f"\nCSV dosyasındaki videolar '{EXPECTED_VIDEO_EXTENSION}' uzantısı ile aranıyor...")

    for index, row in df.iterrows():
        try:
            if 'video_name' not in row or not isinstance(row['video_name'], str) or not row['video_name'].strip():
                continue
            if 'label' not in row or pd.isna(row['label']):
                print(f"UYARI: Satır {index}: Geçersiz veya eksik 'label'. Atlanıyor.")
                continue

            base_filename = row['video_name'].strip()
            label = str(row['label']).strip()

            if not base_filename or not label:
                continue

            if '.' in os.path.basename(base_filename):
                video_filename = base_filename
            else:
                video_filename = f"{base_filename}{EXPECTED_VIDEO_EXTENSION}"

            video_path = os.path.join(VIDEO_DIR, video_filename)

            if os.path.exists(video_path) and os.path.isfile(video_path):
                video_paths_initial.append(video_path)
                string_labels_initial.append(label)
        except Exception as e:
            print(f"HATA: Satır {index} işlenirken: {row}. Hata: {e}")

    print(f"\nToplam {len(video_paths_initial)} geçerli video bulundu (filtrelenmeden önce).")
    if not video_paths_initial:
        raise ValueError("Hiç geçerli video bulunamadı!")

    temp_label_encoder = LabelEncoder()
    temp_encoded_labels = temp_label_encoder.fit_transform(string_labels_initial)
    print(f"Filtrelemeden önce {len(temp_label_encoder.classes_)} adet benzersiz etiket bulundu.")

    unique_temp_labels, counts_temp = np.unique(temp_encoded_labels, return_counts=True)
    valid_temp_indices = unique_temp_labels[counts_temp >= MIN_SAMPLES_PER_CLASS_FOR_STRATIFY]
    valid_string_labels = temp_label_encoder.inverse_transform(valid_temp_indices)

    video_paths_filtered = []
    string_labels_filtered = []

    for i in range(len(string_labels_initial)):
        if string_labels_initial[i] in valid_string_labels:
            video_paths_filtered.append(video_paths_initial[i])
            string_labels_filtered.append(string_labels_initial[i])

    if not video_paths_filtered:
        print(
            f"UYARI: Filtreleme sonrası hiç video kalmadı! Her sınıfta en az {MIN_SAMPLES_PER_CLASS_FOR_STRATIFY} örnek olmalı.")
        return np.array([]), np.array([]), LabelEncoder(), False, 0

    print(
        f"Filtreleme sonrası {len(video_paths_filtered)} video kaldı (sınıf başına en az {MIN_SAMPLES_PER_CLASS_FOR_STRATIFY} örnek).")

    final_label_encoder = LabelEncoder()
    y_final_encoded = final_label_encoder.fit_transform(string_labels_filtered)
    num_classes_final = len(final_label_encoder.classes_)
    print(f"Filtreleme sonrası {num_classes_final} adet benzersiz etiket (sınıf) kaldı.")

    X_final = []
    print(f"\n{len(video_paths_filtered)} filtrelenmiş video için kareler çıkarılıyor...")
    for i, video_path in enumerate(video_paths_filtered):
        if (i + 1) % 10 == 0 or (i + 1) == len(video_paths_filtered):
            print(f"  Video {i + 1}/{len(video_paths_filtered)} işleniyor: {os.path.basename(video_path)}")
        frames = extract_frames(video_path)
        X_final.append(frames)
    print("Kare çıkarma tamamlandı.")

    X_final = np.array(X_final)

    print(f"Filtrelenmiş veri seti X'in şekli: {X_final.shape}")
    print(f"Filtrelenmiş etiket seti y'nin şekli: {y_final_encoded.shape}")

    can_stratify_final = False
    if len(y_final_encoded) > 0:
        unique_check_labels, counts_check = np.unique(y_final_encoded, return_counts=True)
        if len(counts_check) > 0 and counts_check.min() >= MIN_SAMPLES_PER_CLASS_FOR_STRATIFY:
            can_stratify_final = True
            print(f"Filtrelenmiş veri setinde sınıf başına min. örnek: {counts_check.min()}. Stratify kullanılabilir.")
        elif len(counts_check) > 0:
            print(
                f"UYARI: Filtrelenmiş veri setinde sınıf başına min. örnek: {counts_check.min()}. Stratify kullanılamaz.")
        else:
            print("UYARI: Filtrelenmiş veri setinde hiç sınıf bulunamadı.")

    return X_final, y_final_encoded, final_label_encoder, can_stratify_final, num_classes_final

# build_model_for_grid fonksiyonu artık sadece model inşa eder
def build_keras_model(input_shape, num_classes, dense_units=32, learning_rate=0.001, alpha=0.01, dropout_rate=0.4):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(units=dense_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(alpha)))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    start_time = time.time()
    try:
        X, y, label_encoder, can_stratify, num_classes = load_dataset()
        if len(X) == 0:
            print("Filtreleme sonrası hiç veri kalmadığı veya yüklenemediği için işlem durduruluyor.")
            return
        if num_classes == 0:
            print("Hiç sınıf bulunamadığı için işlem durduruluyor.")
            return
    except Exception as e:
        print(f"\nVeri yüklenirken hata: {e}")
        traceback.print_exc()
        return

    print(f"\nVeri başarıyla yüklendi ve filtrelendi. {len(X)} video, {num_classes} sınıf ile devam edilecek.")

    test_set_proportion = 0.20

    min_test_samples_for_stratify = num_classes if can_stratify else 1
    min_train_samples_for_stratify = num_classes if can_stratify else 1

    actual_test_samples = 0

    if len(X) < min_test_samples_for_stratify + min_train_samples_for_stratify:
        print(
            f"UYARI: Toplam örnek sayısı ({len(X)}) stratifikasyonlu düzgün bir bölme için çok az ({num_classes} sınıf var). Stratify kapatılıyor.")
        can_stratify = False
        actual_test_samples = int(np.ceil(len(X) * test_set_proportion))
        if actual_test_samples == 0 and len(X) > 0: actual_test_samples = 1
        if actual_test_samples >= len(X) and len(X) > 0: actual_test_samples = len(X) - 1
        if actual_test_samples == 0:
            print("HATA: Test seti için bile yeterli örnek yok.")
            return

    elif can_stratify:
        calculated_test_samples = int(np.ceil(len(X) * test_set_proportion))
        actual_test_samples = max(calculated_test_samples, num_classes)

        # Test setini ayırdıktan sonra eğitim için yeterli örnek kalmalı (stratify için sınıf sayısı kadar)
        if (len(X) - actual_test_samples) < num_classes and num_classes > 0:
             print(
                 f"UYARI: Test setini ({actual_test_samples} örnek) ayırdıktan sonra eğitime yeterli ({num_classes} gerekirken {len(X) - actual_test_samples} kalıyor) "
                 f"örnek kalmıyor. Test örnek sayısı azaltılıyor veya stratify kapatılıyor.")
             actual_test_samples = len(X) - num_classes
             if actual_test_samples < num_classes: # Azaltmaya rağmen yetmiyorsa
                 print("UYARI: Stratifikasyon için hala yeterli veri yok. Stratify kapatılıyor.")
                 can_stratify = False
                 actual_test_samples = int(np.ceil(len(X) * test_set_proportion)) # Yeniden hesapla
                 if actual_test_samples == 0 and len(X) > 0: actual_test_samples = 1
                 if actual_test_samples >= len(X) and len(X) > 0: actual_test_samples = len(X) - 1


    else: # Stratify yapılamıyorsa
        actual_test_samples = int(np.ceil(len(X) * test_set_proportion))
        if actual_test_samples == 0 and len(X) > 0: actual_test_samples = 1
        if actual_test_samples >= len(X) and len(X) > 0: actual_test_samples = len(X) - 1


    if actual_test_samples >= len(X):
         print(f"HATA: Test seti boyutu ({actual_test_samples}) toplam örnek sayısına ({len(X)}) eşit veya daha büyük olamaz. Test set oranı veya veri setinizi kontrol edin.")
         return
    if (len(X) - actual_test_samples) < (num_classes if can_stratify else 1) and num_classes > 0:
         print(f"HATA: Test setini ayırdıktan sonra eğitim için yeterli örnek kalmıyor ({len(X) - actual_test_samples} kaldı, {num_classes if can_stratify else 1} gerekiyor).")
         return


    try:
        print(f"Veri bölme: can_stratify={can_stratify}, test_samples={actual_test_samples}")
        # Test setini ayır
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=actual_test_samples,
            random_state=42,
            stratify=y if can_stratify else None
        )
        print(f"Veri ilk ayrıldı: Ana Eğitim/Doğrulama ({len(X_train_val)} video), Ana Test ({len(X_test)} video)")

        if not can_stratify:
            print("UYARI: Stratify kullanılamadığı için eğitim/test dağılımı dengesiz olabilir.")

    except ValueError as e:
        print(f"\nEğitim/Test ayırma sırasında hata: {e}")
        print("Veri setinizi ve sınıf dağılımınızı kontrol edin. Stratify devre dışı bırakılarak tekrar deneyebilirsiniz.")
        # Eğer ilk ayırmada stratify hatası olursa, stratify'siz yeniden dene
        if can_stratify:
             can_stratify = False
             actual_test_samples = int(np.ceil(len(X) * test_set_proportion)) # Yeniden hesapla
             if actual_test_samples == 0 and len(X) > 0: actual_test_samples = 1
             if actual_test_samples >= len(X) and len(X) > 0: actual_test_samples = len(X) - 1
             if actual_test_samples >= len(X):
                  print(f"HATA: Test seti boyutu ({actual_test_samples}) toplam örnek sayısına ({len(X)}) eşit veya daha büyük olamaz. Test set oranı veya veri setinizi kontrol edin.")
                  return
             if (len(X) - actual_test_samples) < (num_classes if can_stratify else 1) and num_classes > 0:
                  print(f"HATA: Test setini ayırdıktan sonra eğitim için yeterli örnek kalmıyor ({len(X) - actual_test_samples} kaldı, {num_classes if can_stratify else 1} gerekiyor).")
                  return

             print("Stratify devre dışı bırakılarak yeniden ayırma deneniyor...")
             X_train_val, X_test, y_train_val, y_test = train_test_split(
                 X, y,
                 test_size=actual_test_samples,
                 random_state=42,
                 stratify=None # Stratify kapatıldı
             )
             print(f"Stratify devre dışı bırakılarak veri ayrıldı: Ana Eğitim/Doğrulama ({len(X_train_val)} video), Ana Test ({len(X_test)} video)")


    if len(X_train_val) == 0:
        print("HATA: Eğitim/Doğrulama seti boş. Veri bölme oranlarını kontrol edin.")
        return


    # parametre ızgarasını manuel olarak tanımla
    param_grid = {
        'dense_units': [32, 64],
        'learning_rate': [0.001, 0.0001],
        'alpha': [0.01, 0.1],
        'dropout_rate': [0.3, 0.4],
        'batch_size': [32],
        'epochs': [10]
    }

    param_combinations = list(product(
        param_grid['dense_units'],
        param_grid['learning_rate'],
        param_grid['alpha'],
        param_grid['dropout_rate'],
        param_grid['batch_size'],
        param_grid['epochs']
    ))

    print("\nTanımlanan Parametre Izgarası (Manuel Tarama İçin):")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")

    print(f"\nManuel Hiperparametre Taraması Başlatılıyor ({len(param_combinations)} kombinasyon)...")

    best_mean_val_accuracy = -1
    best_params = None
    results = []

    # Çapraz Doğrulama Ayarları
    n_splits = 2 # GridSearch'te kullandığınız CV sayısıyla aynı
    # Manuel CV için StratifiedKFold kullanmak daha iyi
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Eğer stratify ilk bölmede başarısız olduysa, CV'de de kullanılamaz
    if not can_stratify:
         print(f"UYARI: Stratify devre dışı bırakıldığı için KFold kullanılacak (StratifiedKFold yerine).")
         from sklearn.model_selection import KFold
         kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


    tuning_start_time = time.time()

    for i, (dense_units, learning_rate, alpha, dropout_rate, batch_size, epochs) in enumerate(param_combinations):
        print(f"\nKombinasyon {i+1}/{len(param_combinations)}: "
              f"dense_units={dense_units}, lr={learning_rate}, alpha={alpha}, dropout_rate={dropout_rate}")

        fold_val_accuracies = []

        # Manuel Çapraz Doğrulama Döngüsü
        try:
            # Eğitim/Doğrulama veri setini CV katmanlarına ayır
            split_generator = kf.split(X_train_val, y_train_val) if can_stratify else kf.split(X_train_val)

            for fold_idx, (train_index, val_index) in enumerate(split_generator):
                print(f"  Fold {fold_idx+1}/{n_splits} Eğitiliyor...")
                X_fold_train, X_fold_val = X_train_val[train_index], X_train_val[val_index]
                y_fold_train, y_fold_val = y_train_val[train_index], y_train_val[val_index]

                # Kare verisini yeniden şekillendir
                X_fold_train_reshaped = X_fold_train.reshape(-1, FRAME_SIZE[0], FRAME_SIZE[1], 3)
                y_fold_train_reshaped = np.repeat(y_fold_train, MAX_FRAMES, axis=0)
                X_fold_val_reshaped = X_fold_val.reshape(-1, FRAME_SIZE[0], FRAME_SIZE[1], 3)
                y_fold_val_reshaped = np.repeat(y_fold_val, MAX_FRAMES, axis=0)


                # Modeli oluştur
                model = build_keras_model(
                    input_shape=(FRAME_SIZE[0], FRAME_SIZE[1], 3),
                    num_classes=num_classes,
                    dense_units=dense_units,
                    learning_rate=learning_rate,
                    alpha=alpha,
                    dropout_rate=dropout_rate
                )

                # Early Stopping'i kullan
                early_stopping = EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    verbose=0, # verbose=0 yaparak her epoch çıktısını gizleyin
                    restore_best_weights=True,
                    mode='max'
                )

                # Modeli eğit
                history = model.fit(
                    X_fold_train_reshaped, y_fold_train_reshaped,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_fold_val_reshaped, y_fold_val_reshaped),
                    callbacks=[early_stopping],
                    verbose=0 # verbose=0 yaparak her epoch çıktısını gizleyin
                )

                # Modeli doğrulama verisi üzerinde değerlendir
                # Modelin compile metrics'leri otomatik olarak döndürülecektir
                loss, accuracy = model.evaluate(X_fold_val_reshaped, y_fold_val_reshaped, verbose=0)
                print(f"    Fold {fold_idx+1} Doğruluk: {accuracy:.4f}")
                fold_val_accuracies.append(accuracy)

                # Modeli temizle (hafıza sızıntısını önlemek için)
                del model
                tf.keras.backend.clear_session()

        except Exception as e:
            print(f"\nKombinasyon {i+1} için Fold eğitimi sırasında HATA: {e}")
            traceback.print_exc()
            # Hata durumunda bu kombinasyonu atla
            fold_val_accuracies = [-1] # Hata olduğunu belirtmek için düşük değer atayabiliriz
            pass # Döngüye devam et

        mean_val_accuracy = np.mean(fold_val_accuracies) if fold_val_accuracies and -1 not in fold_val_accuracies else -1
        results.append({
            'params': {
                'dense_units': dense_units,
                'learning_rate': learning_rate,
                'alpha': alpha,
                'dropout_rate': dropout_rate,
                'batch_size': batch_size,
                'epochs': epochs
            },
            'mean_val_accuracy': mean_val_accuracy,
            'fold_val_accuracies': fold_val_accuracies
        })
        print(f"Kombinasyon {i+1} Ortalama Doğruluk (CV): {mean_val_accuracy:.4f}")

        # En iyi parametreleri güncelle
        if mean_val_accuracy > best_mean_val_accuracy:
            best_mean_val_accuracy = mean_val_accuracy
            best_params = results[-1]['params']

    tuning_end_time = time.time()
    print(f"\nManuel Hiperparametre Tarama Tamamlandı. Süre: {(tuning_end_time - tuning_start_time) / 60:.2f} dakika")

    print(f"\nEn iyi Ortalama CV Doğruluğu: {best_mean_val_accuracy:.4f}")
    print("En iyi parametreler:")
    if best_params:
        for param, value in best_params.items():
            print(f"  {param}: {value}")
    else:
        print("Uygun en iyi parametre bulunamadı.")
        return # En iyi parametre bulunamazsa devam etme


    print("\nEn iyi parametrelerle final model eğitiliyor (Tüm Eğitim/Doğrulama Verisi Üzerinde)...")

    try:
        # Tüm eğitim/doğrulama veri setini kullanarak final modeli eğitin
        X_train_val_reshaped = X_train_val.reshape(-1, FRAME_SIZE[0], FRAME_SIZE[1], 3)
        y_train_val_reshaped = np.repeat(y_train_val, MAX_FRAMES, axis=0)

        # En iyi parametrelerle final modeli oluştur
        final_model = build_keras_model(
            input_shape=(FRAME_SIZE[0], FRAME_SIZE[1], 3),
            num_classes=num_classes,
            dense_units=best_params['dense_units'],
            learning_rate=best_params['learning_rate'],
            alpha=best_params['alpha'],
            dropout_rate=best_params['dropout_rate']
        )

        # Early Stopping'i tekrar ekleyin (Doğrulama verisi artık X_val değil, X_train_val içinden bir kısım olabilir,
        # ancak bu durumda X_val'ı ayırmadığımız için validation_split kullanmak daha doğru olabilir
        # veya validation_data olarak X_val'ı manuel olarak tekrar ayırmak gerekir.
        # Basitlik için EarlyStopping'i şimdilik tüm train_val üzerinde kullanıyoruz,
        # veya validation_split parametresini kullanabiliriz.)

        # validation_split kullanarak EarlyStopping için doğrulama seti ayırma
        final_early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            verbose=1,
            restore_best_weights=True,
            mode='max'
        )

        # Modeli tüm eğitim/doğrulama veri seti üzerinde eğit
        final_model.fit(
            X_train_val_reshaped, y_train_val_reshaped,
            epochs=best_params['epochs'] * 2, # Final eğitimde biraz daha fazla epoch verilebilir
            batch_size=best_params['batch_size'],
            validation_split=0.15, # Training/Validation setinden %15 doğrulama için ayır
            callbacks=[final_early_stopping],
            verbose=1
        )
        print("Final model eğitimi tamamlandı.")

    except Exception as e:
        print(f"\nFinal model eğitimi sırasında HATA: {e}")
        traceback.print_exc()
        return


    print("\nFinal model test verisi üzerinde değerlendiriliyor...")

    try:
        X_test_reshaped = X_test.reshape(-1, FRAME_SIZE[0], FRAME_SIZE[1], 3)
        y_test_reshaped = np.repeat(y_test, MAX_FRAMES, axis=0)

        # Final modeli test seti üzerinde değerlendir
        test_loss, test_accuracy = final_model.evaluate(X_test_reshaped, y_test_reshaped, verbose=0)
        print(f"==> Final modelin test kaybı: {test_loss:.4f}")
        print(f"==> Final modelin test doğruluğu: {test_accuracy:.4f} (% {test_accuracy * 100:.2f})")

        # Tahmin yap ve raporları oluştur
        y_pred_classes = np.argmax(final_model.predict(X_test_reshaped), axis=1)

        # Benzersiz tahmin edilen sınıfları al
        unique_pred_classes = np.unique(y_pred_classes)
        # Tahmin edilen sınıflara göre target_names'i filtrele
        # Etiket kodlayıcının tüm sınıflarını kullanmak daha güvenli olabilir
        all_target_names = label_encoder.classes_

        print("\nSınıflandırma Raporu (Test Seti - Kare Bazlı):")
        # labels parametresini y_test_reshaped'deki benzersiz değerleri veya tüm olası sınıfları içerecek şekilde ayarlayabilirsiniz
        # Burada label_encoder.classes_ kullanmak daha geneldir
        print(classification_report(y_test_reshaped, y_pred_classes, labels=np.arange(num_classes), target_names=all_target_names, zero_division=0))

        print("Karmaşıklık Matrisi (Test Seti - Kare Bazlı):")
        # labels parametresini np.arange(num_classes) olarak ayarlayarak tüm sınıflar için matris oluştur
        cm = confusion_matrix(y_test_reshaped, y_pred_classes, labels=np.arange(num_classes))
        fig_width = max(10, num_classes // 1.5 if num_classes > 0 else 10)
        fig_height = max(8, num_classes // 2 if num_classes > 0 else 8)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=all_target_names, yticklabels=all_target_names)
        plt.xlabel('Tahmin Edilen (Predicted)')
        plt.ylabel('Gerçek (True)')
        plt.title('Karmaşıklık Matrisi (Test Seti - Kare Bazlı)')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\nFinal model değerlendirilirken HATA oluştu: {e}")
        traceback.print_exc()


    try:
        # Final modeli ve etiket kodlayıcıyı kaydet
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        model_save_path = os.path.join(MODEL_SAVE_DIR, "manual_tuned_video_classification_model.keras")
        encoder_save_path = os.path.join(MODEL_SAVE_DIR, "label_encoder_classes_manual_tuned.npy")

        if final_model:
            final_model.save(model_save_path)
            print(f"\nFinal model kaydedildi: {model_save_path}")
            np.save(encoder_save_path, label_encoder.classes_)
            print(f"Etiket kodlayıcı sınıfları kaydedildi: {encoder_save_path}")
        else:
            print("Kaydedilecek geçerli bir final model bulunamadı.")

    except Exception as e:
        print(f"\nModel veya etiket kodlayıcı kaydedilirken HATA oluştu: {e}")
        traceback.print_exc()

    end_time = time.time()
    print(f"\nToplam Çalışma Süresi: {(end_time - start_time) / 60:.2f} dakika")


if __name__ == "__main__":
    main()