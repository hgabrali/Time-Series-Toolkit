


# PART 1: 🚂 Loading the Train Data: Step-by-Step Process
*(Eğitim Verisini Yükleme: Adım Adım Süreç)*

Önceki aşamada çoğu dosya başarıyla yüklenmişti, ancak mağaza satışlarının zaman serisi verilerini (*time-series data*) içeren `train.csv` dosyası hariç. Bu dosya oldukça büyüktür ve standart yöntemlerle okunması zordur.

Bu bölümde, büyük veri dosyalarını yönetmek için **"Chunking"** (Parçalara Ayırma) ve **"Sampling"** (Örnekleme) stratejilerini uyguladık. İşte adım adım uygulanan işlemler:

---

### 1. Install the `gdown` Library
*(gdown Kütüphanesinin Kurulumu)*
* **İşlem (Action):** `gdown` kütüphanesini ortamımıza kurduk.
* **Neden? (Why?):** Büyük dosyamız Google Drive üzerinde barınmaktadır. Standart `requests` kütüphanesi yerine, büyük dosyaları indirme (*large file download*) konusunda daha yetenekli ve stabil olan `gdown` aracını tercih ettik.

### 2. Download the `train.csv` File
*(train.csv Dosyasının İndirilmesi)*
* **İşlem (Action):** Dosyayı indirdik ve çalışma ortamımıza `train.csv` adıyla kaydettik.
* **Neden? (Why?):** Veriyi işleyebilmek için yerel ortama (*local environment*) veya Colab diskine alınması gerekiyordu.

### 3. Select Stores from the "Pichincha" Region
*(Pichincha Bölgesinden Mağazaların Seçimi)*
* **İşlem (Action):** Analiz kapsamımızı daraltmak için mağaza listesini filtreledik ve sadece "Pichincha" bölgesindeki mağazaları seçtik.
* **Neden? (Why?):** Tüm veri yerine belirli bir bölgeye odaklanarak analizi daha yönetilebilir hale getirdik (*region filtering*).

### 4. Read the Data in Chunks and Filter by Store
*(Veriyi Parçalar Halinde Okuma ve Mağazaya Göre Filtreleme)*
* **İşlem (Action):** `train.csv` dosyası çok büyük olduğu için tek seferde okumak yerine, 1 milyon satırlık **parçalar halinde** (*chunks*) okuduk. Her parça okunduğunda, sadece yukarıda seçtiğimiz "Pichincha" mağazalarına ait satırları tuttuk.
* **Neden? (Why?):** Bellek yönetimi (*memory management*) için kritiktir. Tüm dosyayı RAM'e yüklemek yerine, parça parça işleyip gereksiz veriyi anında elemek (*on-the-fly filtering*) sistemin çökmesini engeller.

### 5. Combine the Chunks
*(Parçaların Birleştirilmesi)*
* **İşlem (Action):** Filtrelenmiş parçaları (*filtered chunks*) tek bir DataFrame çatısı altında birleştirdik (*concatenation*).
* **Neden? (Why?):** Analiz ve modelleme aşamasında bütüncül bir veri seti üzerinde çalışabilmek için parçaları tekrar bir araya getirmemiz gerekiyordu.

### 6. Sample 2 Million Rows
*(2 Milyon Satırın Örneklenmesi)*
* **İşlem (Action):** Filtrelenmiş veriden rastgele 2 milyon satır seçtik (*random sampling*).
* **Neden? (Why?):** Eğitim amaçlı (*educational sake*) hesaplamaları hızlandırmak için. Gerçek dünyada tüm veri kullanılabilir ancak öğrenme sürecinde işlem süresini kısaltmak (*computation speed*) için veri boyutu optimize edildi.

### 7. Clean Up
*(Temizlik)*
* **İşlem (Action):** Bellekte yer kaplayan geçici parça listelerini sildik.
* **Neden? (Why?):** Python'ın bellek yönetimini rahatlatmak (*garbage collection*) ve RAM'i verimli kullanmak için gereksiz değişkenler temizlendi.

### 8. Verification
*(Doğrulama)*
* **İşlem (Action):** Oluşturulan DataFrame'lerin ilk satılarını (`head`) yazdırarak kontrol ettik.
* **Neden? (Why?):** Verinin düzgün yüklenip yüklenmediğini ve formatın beklediğimiz gibi olup olmadığını teyit etmek (*sanity check*).

---

# PART 2: 
