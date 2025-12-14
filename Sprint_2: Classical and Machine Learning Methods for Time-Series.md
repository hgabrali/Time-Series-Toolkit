# 📅  Classical and Machine Learning Methods for Time-Series
*(Hafta 2: Zaman Serileri için Klasik ve Makine Öğrenimi Yöntemleri)*

> **Date Range:** Dec 8 - Dec 15

In this module, we will explore both **classical time-series methods** (*klasik zaman serisi yöntemleri*) and **machine learning approaches** (*makine öğrenimi yaklaşımları*) for forecasting. We aim to bridge the gap between traditional statistical theory and modern predictive algorithms.
*(Bu modülde, tahminleme için hem klasik zaman serisi yöntemlerini hem de makine öğrenimi yaklaşımlarını inceleyeceğiz. Geleneksel istatistiksel teori ile modern tahmin algoritmaları arasındaki boşluğu doldurmayı hedefliyoruz.)*

---

## 🎯 Learning Objectives & Key Outcomes
*(Öğrenme Hedefleri ve Temel Çıkarımlar)*

By the end of this week, you will be able to:
*(Bu haftanın sonunda şunları yapabileceksiniz:)*

### 1. 📉 Classical Time-Series Modelling
*(Klasik Zaman Serisi Modellemesi)*
* **Implement models:** Build and tune classical statistical models like **ARIMA** (*AutoRegressive Integrated Moving Average*) and **SARIMA** (*Seasonal ARIMA*).
    *(ARIMA ve SARIMA gibi klasik istatistiksel modelleri kurma ve ayarlama.)*
* **Core Concepts:** Understand **stationarity** (*durgunluk*), **differencing** (*fark alma*) and **seasonality** (*mevsimsellik*) handling in linear models.
    *(Lineer modellerde durgunluk, fark alma ve mevsimsellik yönetimini anlama.)*

### 2. 🌲 Machine Learning Approaches
*(Makine Öğrenimi Yaklaşımları)*
* **Apply Tree-Based Models:** Utilize powerful algorithms like **XGBoost**, **LightGBM**, or **Random Forest** for time-series forecasting.
    *(Zaman serisi tahmini için XGBoost, LightGBM veya Random Forest gibi güçlü algoritmaları kullanma.)*
* **Handling Non-Linearity:** Learn how these models capture non-linear relationships better than traditional methods.
    *(Bu modellerin doğrusal olmayan ilişkileri geleneksel yöntemlerden nasıl daha iyi yakaladığını öğrenme.)*

### 3. 🧠 Deep Learning Foundations
*(Derin Öğrenme Temelleri)*
* **Introduction to Neural Networks:** Get familiar with deep learning approaches tailored for sequential data, specifically:
    *(Sıralı veriler için özelleştirilmiş derin öğrenme yaklaşımlarına aşina olma, özellikle:)*
    * **RNNs** (*Recurrent Neural Networks - Tekrarlayan Sinir Ağları*)
    * **LSTMs** (*Long Short-Term Memory - Uzun Kısa Süreli Bellek Ağları*)
* **Sequence Modeling:** Understand how these networks manage **long-term dependencies** (*uzun vadeli bağımlılıklar*) in time series.

### 4. 🛠️ Advanced Feature Engineering
*(İleri Seviye Özellik Mühendisliği)*
* **Data Preprocessing:** Perform preprocessing specifically tailored for supervised machine learning models.
    *(Gözetimli makine öğrenimi modelleri için özel olarak uyarlanmış ön işleme gerçekleştirme.)*
* **Creating Signals:** Generate powerful features such as:
    *(Güçlü özellikler üretme:)*
    * **Lag Features** (*Gecikmeli Özellikler*)
    * **Rolling Window Statistics** (*Kayan Pencere İstatistikleri*)
    * **Time-Based Components** (*Zaman Bazlı Bileşenler - Gün, Ay, Yıl vb.*)

### 5. ⚖️ Strategic Comparison
*(Stratejik Karşılaştırma)*
* **Critical Analysis:** Understand the **differences**, **benefits**, and **challenges** of classical statistical methods versus machine learning approaches.
    *(Klasik istatistiksel yöntemler ile makine öğrenimi yaklaşımları arasındaki farkları, faydaları ve zorlukları anlama.)*
* **Decision Making:** Learn when to use which method based on data size, interpretability requirements, and computational resources.
    *(Veri boyutu, yorumlanabilirlik gereksinimleri ve hesaplama kaynaklarına bağlı olarak hangi yöntemin ne zaman kullanılacağını öğrenme.)*

-------
-------
# 🎯 Preparing Data & Introduction to Darts
*(Veri Hazırlama ve Darts'a Giriş)*

Modelleme aşamasına geçmeden önce, üzerinde çalışmaya hazır, temiz bir veri setine ihtiyacımız vardır. Bu bölümde, klasik **ARIMA/SARIMA** modellerini uygulamak için Python'un güçlü zaman serisi kütüphanesi **Darts**'ı kullanacağız. Veri seti olarak yine "Corporación Favorita Grocery Sales Forecasting" verilerini kullanıyoruz.

---

## 🧐 What is DARTS? Why use it?

**DARTS**, zaman serisi tahmini (*time-series forecasting*) için özel olarak tasarlanmış, kullanımı kolay ve üst düzey (*high-level*) bir Python kütüphanesidir.

> **💡 The Core Philosophy (Temel Felsefe):**
> DARTS, `scikit-learn` kütüphanesinin kullanıcı dostu yapısını (fit/predict mantığı) zaman serilerine uyarlar. Karmaşık modelleri (ARIMA'dan Derin Öğrenme Transformer'larına kadar) tek bir satır kodla değiştirmenize ve test etmenize olanak tanır.

### Key Features (Temel Özellikler)
1.  **Unified API (Birleşik Arayüz):** Klasik istatistiksel modellerden (*ARIMA, Exponential Smoothing*) modern makine öğrenimi (*XGBoost, LightGBM*) ve derin öğrenme modellerine (*N-BEATS, LSTM, Transformers*) kadar her şeyi aynı arayüzle kullanabilirsiniz.
2.  **Multivariate Support (Çok Değişkenli Destek):** Sadece hedef değişkeni değil, dışsal faktörleri (*past/future covariates*) de modele kolayca dahil edebilirsiniz (örn. Tatiller, Petrol Fiyatları).
3.  **Backtesting & Evaluation (Geriye Dönük Test ve Değerlendirme):** Modelin geçmiş performansını simüle etmek için güçlü araçlar sunar.
4.  **Probabilistic Forecasting (Olasılıksal Tahmin):** Sadece tek bir değer değil, güven aralıkları (*confidence intervals*) ve olasılık dağılımları üretebilir.

---

## 🛠️ Step-by-Step Workflow


### Step 0: Installing Darts
*(Adım 0: Darts Kurulumu)*

We will illustrate how to use classical time-series methods like ARIMA for forecasting using the DARTS library. First, ensure the library is installed in your environment.
*(DARTS kütüphanesini kullanarak tahminleme için ARIMA gibi klasik yöntemlerin nasıl kullanılacağını göstereceğiz. Öncelikle kütüphanenin ortamınızda kurulu olduğundan emin olun.)*

### Step 1: Loading a Single Store–Item Series
*(Adım 1: Tek Bir Mağaza-Ürün Serisini Yükleme)*

Rather than juggling millions of rows, we’ll extract **one product in one store** and truncate everything after March 31, 2014.
*(Milyonlarca satırla boğuşmak yerine, tek bir mağazadaki tek bir ürünü çıkaracağız ve 31 Mart 2014'ten sonraki verileri keseceğiz.)*

* **Goal (Hedef):** Create a manageable dataset to learn the mechanics of the model.
* **Action (Eylem):** Filter by `store_nbr`, `item_nbr`, and apply `date < '2014-04-01'`.
* **Result (Sonuç):** A tidy DataFrame with just daily sales for our chosen series. (*Seçilen serimiz için sadece günlük satışları içeren düzenli bir DataFrame.*)

### Step 2: Prepare & Convert to `TimeSeries`
*(Adım 2: Hazırlık ve `TimeSeries` Nesnesine Dönüştürme)*

This is the most critical step specific to Darts. Darts works with its own object type called `TimeSeries`, not standard Pandas DataFrames.
*(Bu, Darts'a özgü en kritik adımdır. Darts, standart Pandas DataFrame'leri ile değil, `TimeSeries` adı verilen kendi nesne türüyle çalışır.)*

**The Process:**
1.  **Datetime Conversion:** Convert the `date` column to datetime objects and set it as the index.
    *(Tarih sütununu datetime nesnelerine çevirin ve indeks olarak ayarlayın.)*
2.  **Aggregation:** Aggregate by date (summing all `unit_sales` for that day).
    *(Tarihe göre toplulaştırma yapın [o günkü tüm birim satışları toplayarak].)*
3.  **Gap Filling (Reindexing):** Reindex to a complete daily calendar, filling any missing dates with **zero**.
    *(Eksik tarihleri sıfır ile doldurarak tam bir günlük takvime göre yeniden indeksleyin.)*
4.  **Darts Conversion:** Finally, wrap it in Darts’ `TimeSeries` object using `TimeSeries.from_dataframe()`.
    *(Son olarak, `TimeSeries.from_dataframe()` kullanarak veriyi Darts TimeSeries nesnesine sarın.)*

> **✅ Why?** This ensures all the library’s modeling and backtesting tools work **out of the box** (*kurulum gerektirmeden/hazır olarak*).

---

### 📊 Visual Analysis & Interpretation
*(Görsel Analiz ve Yorumlama)*



#### 💡 Think First!


Before reading our interpretation, take a moment to reflect on the chart yourself:
*(Yorumumuzu okumadan önce, grafik üzerinde düşünmek için bir dakikanızı ayırın:)*
* What do you notice about the **sales volume**? Is it consistent, volatile, or trending?
* Do the **spikes** (*sıçramalar*) follow any visible pattern?
* What kinds of **features** might help the model capture this behavior?

#### 📉 Our Analysis


**1. Key Observations (Temel Gözlemler)**
* **Consistently High Sales:** The product sells every day, typically between 300 and 800 units. It is a **high-volume item**.
* **No Missing/Zero Values:** We see activity on every day. This is ideal for training a model that learns from historical behavior without needing complex imputation (*eksik veri doldurma*).
* **Frequent, Sharp Fluctuations:** The series is **noisy** (*gürültülü*) — it goes up and down regularly — but mostly within a predictable range.
* **Occasional Large Peaks:** Some spikes rise sharply (>1000) above the usual range. These likely correspond to **promotions**, **holidays**, or **special events**.

**2. What This Suggests for Forecasting (Tahminleme İçin Ne Anlama Geliyor?)**
* **Lag Features:** The model will benefit from lags (e.g., `sales_lag_1`, `rolling_mean_7`) to learn local dynamics.
* **Calendar Features:** Including `day_of_week`, `is_weekend`, or `month` will help capture recurring patterns (*tekrarlayan kalıplar*).
* **Volatility Handling:** Applying a **rolling average** or using a **log transformation** might improve model stability against noise.
* **External Signals:** To predict the huge spikes, adding **promotions** data (as *covariates*) would be valuable.

---

### Step 3: Splitting the Data into Training and Testing Sets
*(Adım 3: Veriyi Eğitim ve Test Setlerine Ayırma)*

In time-series forecasting, we cannot use random splitting (like `train_test_split` in sklearn) because the **order of data matters**. We must split chronologically.
*(Zaman serisi tahmininde, rastgele bölme kullanamayız çünkü verinin sırası önemlidir. Kronolojik olarak bölmemiz gerekir.)*

* **Training Set (Eğitim Seti):** The past data used to teach the model patterns (e.g., usually the first 80-90% of the timeline).
    *(Model kalıplarını öğretmek için kullanılan geçmiş veriler.)*
* **Validation/Test Set (Doğrulama/Test Seti):** The recent data used to evaluate how well the model predicts the future (the remaining 10-20%).
    *(Modelin geleceği ne kadar iyi tahmin ettiğini değerlendirmek için kullanılan son veriler.)*

**Darts Method:**
We use specific Darts methods (like `.split_before()`) to ensure no **data leakage** (*veri sızıntısı*) occurs—meaning the model never sees the "future" it is trying to predict during training.


