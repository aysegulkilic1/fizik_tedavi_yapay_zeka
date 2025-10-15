# Görüntü İşleme ve Derin Öğrenme Tabanlı Fizik Tedavi Egzersizleri Değerlendirme Mobil Uygulaması

Bu proje, lisans tezi kapsamında geliştirdiğim, hastaların fizik tedavi egzersizlerini evde doğru bir şekilde yapmalarını sağlayan bir yapay zeka ve mobil uygulama sistemidir.

## Proje Yapısı

Bu repository, projenin iki ana bileşenini içerir:

1.  **`/gercek-zamanli-analiz`**: Bu klasör, `MediaPipe` kütüphanesi kullanılarak geliştirilen ve hastanın vücut hareketlerini anlık olarak analiz eden Python kodlarını içerir. Egzersiz doğru yapıldığında, sistem sayacı artırır ve kullanıcıya anlık geri bildirim sağlar.
2.  **`/makine-ogrenmesi-modeli`**: Bu klasör, tarafımca oluşturulan etiketli bir video veri seti ile eğitilmiş **Evrişimli Sinir Ağı (CNN)** modelini içerir. Model, `TensorFlow`, `Keras` ve `scikit-learn` gibi kütüphaneler kullanılarak eğitilmiş ve başarısını artırmak için manuel hiperparametre optimizasyonu uygulanmıştır.

## 🛠️ Kullanılan Teknolojiler

* **Gerçek Zamanlı Analiz:** Python, OpenCV, MediaPipe
* **Makine Öğrenmesi Modeli:** Python, TensorFlow, Keras, Scikit-learn, Pandas, NumPy

## ✨ Öne Çıkan Başarı

Bu bütünsel çalışma, akademik başarısı sayesinde uluslararası **GITMA 2025 Konferansı’nda** sunulmuştur.
