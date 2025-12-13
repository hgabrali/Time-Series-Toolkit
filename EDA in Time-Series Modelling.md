# 📉 Zaman Serisi Analizi: Ayrıştırma ve Bileşenler (Time Series Decomposition)

Zaman serisi analizinde **"Decomposition" (Ayrıştırma)**, karmaşık bir sinyali, onu oluşturan temel bileşenlere ayırma işlemidir. Bir zaman serisi ($Y_t$) genellikle şu formülle ifade edilir:

$$Y_t = Trend_t + Seasonality_t + Residual_t$$

İşte bu bileşenlerin teknik analizi ve kullanılan modern araçlar:

---

## 1. Structure (Yapı) & Stationarity (Durağanlık)

**Durağanlık**, bir serinin istatistiksel özelliklerinin (ortalama, varyans, otokorelasyon) zaman içinde sabit kalmasıdır. Modellerin çoğu (özellikle **ARIMA** gibi lineer modeller), verinin durağan olduğu varsayımıyla çalışır.

* **Teknik Açıklama:** Eğer seride zamanla değişen bir ortalama (**mean**) veya varyans (**variance**) varsa, seri **"Non-Stationary"** (Durağan Değil) olarak adlandırılır. Bu durum, modelin geleceği tahmin ederken geçmişteki katsayıları yanlış kullanmasına neden olur.

### 🛠 Çözüm Yöntemleri
* **Differencing (Fark Alma):** Seriyi durağanlaştırmak için bir önceki zaman adımından çıkarırız ($y_t - y_{t-1}$).
* **Log Transformation (Logaritmik Dönüşüm):** Varyansı sabitlemek için (**Heteroskedastisiteyi** gidermek) kullanılır.

### 🧰 Kullanılan Tool & Testler
* **Görsel Kontrol:** `matplotlib` veya `seaborn` ile **Line Plot** çizilerek trendin varlığı gözlemlenir.
* **İstatistiksel Test:** **Augmented Dickey-Fuller (ADF) Testi** standarttır.
    * $p\text{-value} < 0.05$ ise seri **Durağandır (Stationary)**.
    * $p\text{-value} > 0.05$ ise seri **Durağan Değildir (Unit Root vardır)**.
* **KPSS Testi:** ADF'nin tamamlayıcısı olarak, serinin trend durağanlığını test eder.

---

## 2. Trend (Eğilim)

**Trend**, verinin uzun vadede yukarı (**Uptrend**) veya aşağı (**Downtrend**) yönlü hareketidir.

* **Teknik Açıklama:** Trend, serinin **"Low Frequency"** (Düşük Frekanslı) bileşenidir. Gürültüden (**Noise**) ve mevsimsellikten arındırıldığında geriye kalan ana yöndür. Deterministik (matematiksel bir formüle uyan) veya Stokastik (rastgele yürüyüş içeren) olabilir.

### 🛠 Çözüm Yöntemleri
* **Smoothing (Yumuşatma):** Hareketli ortalamalar (**Rolling Mean**) kullanılarak kısa vadeli dalgalanmalar filtrelenir ve trend ortaya çıkarılır.
* **Detrending (Trendden Arındırma):** Eğer amaç durağanlık ise, tespit edilen trend seriden matematiksel olarak çıkarılır.

### 🧰 Kullanılan Tool & Algoritmalar
* **Moving Averages (MA):** Basit (**SMA**) veya Üstel (**EMA**) hareketli ortalamalar.
* **Hodrick-Prescott (HP) Filter:** Makroekonomik verilerde trendi döngüden (**cycle**) ayırmak için kullanılır.
* **Mann-Kendall Trend Test:** Trendin istatistiksel olarak anlamlı olup olmadığını test eden parametrik olmayan bir testtir.

---

## 3. Seasonality (Mevsimsellik)

Verinin belirli ve sabit periyotlarla (haftanın günü, yılın ayı gibi) tekrarlayan kalıplar sergilemesidir.

* **Teknik Açıklama:** Mevsimsellik, takvim etkisiyle (**Calendar Effects**) oluşur. Döngüsel (**Cyclical**) hareketlerden farkı, periyodun sabit olmasıdır (Döngüler, örneğin ekonomik krizler, düzensiz aralıklarla olur).

### 🛠 Çözüm Yöntemleri
* **Seasonal Decomposition:** Veriyi **"Additive"** (Toplamsal: $Y = T + S + R$) veya **"Multiplicative"** (Çarpımsal: $Y = T \times S \times R$) olarak ayrıştırmak.
* **Fourier Transform:** Karmaşık mevsimsellikleri (örneğin hem haftalık hem yıllık) sinüs ve kosinüs dalgaları ile modellemek.

### 🧰 Kullanılan Tool & Kütüphaneler
* **Statsmodels (`seasonal_decompose`):** Klasik ayrıştırma için.
* **STL Decomposition (Seasonal-Trend decomposition using LOESS):** Gürültüye karşı daha dayanıklı ve esnek bir yöntemdir.
* **ACF Plot (Autocorrelation Function):** Lag (gecikme) grafiklerinde belirli periyotlarda (örn: her 7. lag'da) zirveler (**spikes**) görülmesi mevsimselliğin en net kanıtıdır.

---

## 4. Anomalies (Anomaliler)

Beklenen modelin (**Trend + Seasonality**) çok dışında kalan, nadir ve açıklanması zor veri noktalarıdır.

* **Teknik Açıklama:** **"Outlier"** (Aykırı Değer) olarak da bilinir. İki türü vardır:
    * **Point Anomaly:** Tek bir noktanın sapması (Örn: Sistemin anlık çökmesi).
    * **Contextual Anomaly:** Noktanın kendisi normal olsa da, bulunduğu zaman dilimine göre anormal olması (Örn: Yaz ortasında kar yağması).

### 🛠 Çözüm Yöntemleri
* **Winsorization / Trimming:** Aykırı değerleri belirli bir persentile (örn: %99) sabitlemek veya silmek.
* **Interpolation:** Silinen anomalilerin yerini, komşu verilerin ortalamasıyla doldurmak.

### 🧰 Kullanılan Tool & Algoritmalar
* **Z-Score:** $|Z| > 3$ olan noktalar anomali kabul edilir.
* **Isolation Forest:** Çok boyutlu verilerde anomalileri izole etmek için kullanılan bir ağaç tabanlı algoritmadır.
* **Prophet:** Facebook'un geliştirdiği bu kütüphane, anomalilere ve tatil günlerine (**Holidays**) karşı oldukça dayanıklıdır ve bunları parametre olarak yönetebilir.
