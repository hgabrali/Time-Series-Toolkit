


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

# PART 2: Veri Setini Anlamak (Understanding the Dataset)

İndirdiğimiz **Corporación Favorita Grocery Sales Forecasting** veri setine yakından bakalım.

### 📂 Girdi Verileri (Input Data)

Çalışacağımız birden fazla `csv` dosyası bulunmaktadır. Bunlar şunları içerir:

#### 1. `train.csv`
* Hedef değişken (*target*) olan `unit_sales` (birim satışlar) verisini `date` (tarih) bazında içerir. Ayrıca `store_nbr` (mağaza no), `item_nbr` (ürün no) ve satırları etiketlemek için benzersiz bir `id` sütunu bulunur.
* Hedef `unit_sales`, tamsayı (*integer*) (örn: bir paket cips) veya ondalıklı sayı (*float*) (örn: 1.5 kg peynir) olabilir.
* `unit_sales` değerinin negatif olması, o ürünün iade edildiğini (*returns*) gösterir.
* `onpromotion` sütunu, o `item_nbr`'ın belirtilen `date` ve `store_nbr` için promosyonda olup olmadığını belirtir.
* Bu dosyadaki `onpromotion` değerlerinin yaklaşık %16'sı `NaN` (eksik veri)'dır.

> ☝🏼 **NOT (NOTE):** Eğitim verileri (*training data*), bir mağaza/tarih kombinasyonu için sıfır `unit_sales` olan ürünlere ait satırları içermez. Ürünün o tarihte mağazada stokta olup olmadığına (*in stock*) dair bir bilgi yoktur ve ekiplerin bu durumu ele almanın en iyi yoluna karar vermesi gerekecektir. Ayrıca, eğitim verilerinde görülen ancak test verilerinde (*test data*) görülmeyen az sayıda ürün vardır.

#### 2. `stores.csv`
* Mağaza üst verilerini (*metadata*) içerir: `city` (şehir), `state` (eyalet), `type` (tür) ve `cluster` (küme).
* `cluster`, benzer mağazaların bir gruplandırmasıdır.

#### 3. `items.csv`
* Ürün üst verilerini (*metadata*) içerir: `family` (aile/kategori), `class` (sınıf) ve `perishable` (bozulabilir).

> ☝🏼 **NOT (NOTE):** `perishable` (bozulabilir) olarak işaretlenen ürünlerin skor ağırlığı (*score weight*) **1.25**'tir; diğerlerinin ağırlığı ise **1.0**'dır.

#### 4. `transactions.csv`
* Her bir `date` ve `store_nbr` kombinasyonu için satış işlemlerinin (*sales transactions*) sayısını içerir. Sadece eğitim verisi zaman aralığı (*training data timeframe*) için dahildir.

#### 5. `oil.csv`
* Günlük petrol fiyatı (*Daily oil price*). Hem eğitim (*train*) hem de test (*test*) verisi zaman aralığındaki değerleri içerir. Ekvador petrol bağımlı (*oil-dependent*) bir ülkedir ve ekonomik sağlığı petrol fiyatlarındaki şoklara karşı oldukça kırılgandır (*highly vulnerable*).

#### 6. `holidays_events.csv`
* Tatiller ve Etkinlikler ile bunlara ait üst veriler.
* `Additional` (Ek) tatiller, normal bir takvim tatiline eklenen günlerdir; örneğin Noel civarında tipik olarak gerçekleşen durumlar gibi (Noel Arifesini tatil yapmak).

> ☝🏼 **NOT (NOTE):** `transferred` (aktarılan) sütununa özellikle dikkat edin. `transferred` olan bir tatil resmi olarak o takvim gününe denk gelir, ancak hükümet tarafından başka bir tarihe taşınmıştır. Bir `transferred` gün, tatilden ziyade normal bir gün gibidir. Aslında kutlandığı günü bulmak için, `type` sütununun `Transfer` olduğu ilgili satıra bakın.
>
> *Örneğin:* `Independencia de Guayaquil` tatili 2012-10-09'dan 2012-10-12'ye aktarılmıştır (*transferred*), yani 2012-10-12'de kutlanmıştır.
>
> `type` değeri `Bridge` (Köprü) olan günler, bir tatile eklenen ekstra günlerdir (örn: tatili uzun bir hafta sonuna uzatmak için). Bunlar genellikle, `Bridge` gününü telafi etmek (*payback*) amacıyla normalde çalışma günü olmayan (örn: Cumartesi) bir günün çalışıldığı `Work Day` (Çalışma Günü) tipi ile telafi edilir.

---

### 📝 Ek Notlar (Additional Notes)

1.  **Maaşlar (Wages):** Kamu sektöründeki maaşlar iki haftada bir, ayın **15'inde** ve **son gününde** ödenir. Süpermarket satışları bundan etkilenebilir.
2.  **Deprem (Earthquake):** 16 Nisan 2016'da Ekvador'da 7.8 büyüklüğünde bir deprem meydana gelmiştir. İnsanlar su ve diğer temel ihtiyaç ürünlerini bağışlayarak yardım çalışmalarında (*relief efforts*) bir araya gelmiş, bu durum depremden sonraki birkaç hafta boyunca süpermarket satışlarını büyük ölçüde etkilemiştir.

---

# PART 3:
