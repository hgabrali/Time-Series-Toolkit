# 📈 Time Series (Zaman Serisi) Analizi

Time Series (Zaman Serisi) analizi veri biliminin en "nazlı" ama en güçlü alanlarından biridir. Standart veri setlerinden (örneğin bir ev fiyatı tahminindeki tablolardan) çok farklı matematiksel ve istatistiksel kuralları vardır.

---

## 🧠 Bölüm 1: Time Series Modelling

Verilen metin, zaman serilerinin temel felsefesini ve bir proje döngüsünü (pipeline) anlatıyor. Bir uzman olarak satır aralarını şöyle okumalıyız:

### 1. Zaman Serisini "Benzersiz" Yapan Nedir? (The Unique Nature)

Standart Makine Öğrenmesi (Machine Learning) algoritmaları genellikle verilerin **I.I.D.** (Independent and Identically Distributed) olduğunu varsayar. Yani satırların birbirinden bağımsız olduğunu düşünür.

> **Time Series Farkı:** Burada veriler **birbirine bağımlıdır**. Bugünün hisse senedi fiyatı, dünün fiyatıyla doğrudan ilişkilidir (Auto-correlation).

* **Temporal Dependencies (Zaman Bağımlılıkları):** Geçmiş, geleceği şekillendirir. Modellerimiz bu "hafızayı" korumak zorundadır.

#### Temel Bileşenler:

* **Trend:** Verinin uzun vadede yukarı veya aşağı yönlü hareketi.
* **Seasonality (Mevsimsellik):** Belirli periyotlarda (günlük, haftalık, yıllık) tekrarlayan desenler. (Örn: Dondurma satışlarının yazın artması).
* **Stationarity (Durağanlık):** Çoğu klasik model (ARIMA gibi), verinin ortalamasının ve varyansının zamanla değişmemesini ister. Veri durağan değilse, onu durağan hale getirmek (Differencing) gerekir.

<img width="892" height="420" alt="image" src="https://github.com/user-attachments/assets/0ba18a14-a41c-4f8d-998f-ebd30da650df" />

---

### 2. Endüstriyel Kullanım Alanları ve Yöntemler (Derinlemesine Bakış)

Metinde geçen sektörlerde, zaman serisi şu kritik soruları çözer:

* **Finance (Finans):** Sadece fiyat tahmini değil, "Volatilite (Oynaklık)" tahmini yapılır.
    * *Yöntem:* `GARCH` modelleri, `LSTM` ağları.
* **Energy (Enerji):** Şebeke dengesi için hayati önem taşır. Üretilen elektriğin anında tüketilmesi gerekir.
    * *Yöntem:* `SARIMA` (Mevsimsellik güçlüdür), `Prophet`.
* **Healthcare (Sağlık):** EKG sinyalleri aslında milisaniyelik zaman serileridir. Anormallik tespiti (Anomaly Detection) burada hayat kurtarır.
* **Cybersecurity:** Ağ trafiğindeki ani "spike"lar (sıçramalar) DDoS saldırısı olabilir.
    * *Yöntem:* `Isolation Forest`, `Autoencoders`.


## 3. Karşılaştırmalı Yöntemler Tablosu (Model Comparison Matrix)

Metinde Sprint 2'de geçen modellerin teknik karşılaştırmasını senin için hazırladım:

| Model Türü | Örnekler | Ne Zaman Kullanılır? | Avantajı | Dezavantajı |
| :--- | :--- | :--- | :--- | :--- |
| **Klasik İstatistiksel** | **ARIMA, SARIMA** | Veri seti küçükse, mevsimsellik netse ve açıklanabilirlik (neden bu sonucu verdi?) önemliyse. | Hızlıdır, az veri ile çalışır, matematiksel temeli sağlamdır. | Karmaşık, doğrusal olmayan (non-linear) ilişkileri yakalayamaz. |
| **Makine Öğrenmesi (ML)** | **XGBoost, LightGBM, Random Forest** | Elimizde sadece zaman değil, dış faktörler (hava durumu, tatil günleri vb.) de varsa. | Çok güçlüdür, karmaşık ilişkileri çözer, şu an endüstri standardıdır. | Gelecekteki trendi (extrapolation) yakalamakta zorlanır (verinin aralığı dışına çıkamaz). |
| **Derin Öğrenme (DL)** | **RNN, LSTM, Transformers** | Veri seti devasaysa (Big Data) ve çok uzun vadeli karmaşık desenler varsa. | Uzun vadeli bağımlılıkları (Long-term dependencies) harika yakalar. | Eğitmesi çok uzun sürer, çok fazla veriye ihtiyaç duyar (Data hungry). |


