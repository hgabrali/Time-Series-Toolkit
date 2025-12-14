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

---
---

# 🕰️ Classical Time-Series Methods: ARIMA & Parameter `d`
*(Klasik Zaman Serisi Yöntemleri: ARIMA ve d Parametresi)*

**ARIMA** (*AutoRegressive Integrated Moving Average*), güçlü mevsimsel kalıpları olmayan zaman serisi verilerini anlamak ve tahmin etmek için kullanılan temel bir modeldir.

---

## 🧐 How Does ARIMA Work?
*(ARIMA Nasıl Çalışır?)*

Imagine you’re trying to forecast the daily sales in a grocery store. ARIMA helps predict the sales for tomorrow by combining three components:
*(Bir marketin günlük satışlarını tahmin etmeye çalıştığınızı hayal edin. ARIMA, üç bileşeni birleştirerek yarının satışlarını tahmin etmeye yardımcı olur:)*

1.  **AR (AutoRegression):** Looking at how past days’ sales have behaved.
    *(Geçmiş günlerin satışlarının nasıl davrandığına bakmak [Otokorelasyon].)*
2.  **I (Integrated):** Correcting for any trends via differencing.
    *(Fark alma yoluyla trendleri düzeltmek [Entegrasyon].)*
3.  **MA (Moving Average):** Learning from the errors made in previous predictions.
    *(Önceki tahminlerde yapılan hatalardan öğrenmek [Hareketli Ortalama].)*

> ☝🏼 **Summary:** ARIMA tries to use the **past values** (AR) and **past prediction errors** (MA) to make accurate forecasts, while adjusting for **trends** in the data (I).

---

## 📉 The "I" in ARIMA and Parameter `d`
*(ARIMA'daki "I" ve d Parametresi)*

Sometimes data isn’t "**stationary**," meaning it has a trend that increases or decreases over time.
*(Bazen veriler "durgun" değildir, yani zamanla artan veya azalan bir trende sahiptir.)*

* **The Integrated (I) part** helps by removing these trends to make the data easier to predict.
* **Method:** It does this by **differencing**—subtracting the previous value from the current value to smooth out the trend.
    *(Yöntem: Bunu fark alma yoluyla yapar—trendi yumuşatmak için önceki değeri mevcut değerden çıkarır.)*



> **Definition:** The `d` in `ARIMA(p,d,q)` tells us **how many times** to difference the series to remove a trend and achieve stationarity (constant mean & variance).

---

## 🛠️ Step-by-Step Guide: Choosing `d`
*(Adım Adım Rehber: d Seçimi)*

### Step 1: Start with No Differencing (`d=0`)
*(Adım 1: Fark Alma Olmadan Başla)*

#### 1. Visual Check: Plot Your Raw Series
*(Görsel Kontrol: Ham Seriyi Çizdir)*
* **Visual Patterns & Drift:** Does a gentle upward or downward drift hide underneath the spikes?
    *(Sıçramaların altında hafif bir yukarı veya aşağı sürüklenme gizleniyor mu?)*
* **Mean Level:** Does it look like the mean level is roughly constant, or does it trend up/down?
    *(Ortalama seviye kabaca sabit mi görünüyor, yoksa yukarı/aşağı trend mi var?)*
* **Stationarity Assessment:** Based on your visual impression, would you call this series “stationary”?

> **Our Analysis (Example):**
> * **Visual:** The series shows strong short-term fluctuations but no clear long-term trend. The mean level appears fairly stable around 500–600 units/day.
> * **Conclusion:** This series appears visually stationary, or at least close enough for many forecasting models (d=0).

#### 2. Rolling Mean Check
*(Hareketli Ortalama Kontrolü)*
Let’s smooth out the day-to-day spikes with a **30-day rolling average**. By averaging over a full month, the noisy zero-and-spike pattern flattens out, revealing whether there really is a gradual trend.



> **Analysis:** The mean level drops early in 2013 but stays stable (430–480 units) afterwards. It may be treated as **near-stationary**.

#### 3. Statistical Check: ADF Test
*(İstatistiksel Kontrol: ADF Testi)*

**What is ADF Test?** (*Augmented Dickey-Fuller Test*)
It is a statistical test used to check for **stationarity**.
* **Null Hypothesis ($H_0$):** The series has a unit root (it is **not stationary**).
* **Alternative Hypothesis ($H_1$):** The series has no unit root (it is **stationary**).

**Decision Rule:**
* If **p-value < 0.05**: Reject $H_0$ → Series is **Stationary**. (Accept `d=0`)
* If **p-value ≥ 0.05**: Fail to reject $H_0$ → Series is **Non-Stationary**. (Try `d=1`)

```python
# Code snippet for ADF Test
# Result Example:
# p-value: 0.000467...
```

## 📉 Step 2: If Not Stationary, Try One Difference (`d=1`)
*(Adım 2: Durgun Değilse, Bir Fark Almayı Dene)*

If the visual check showed a **trend** or the **ADF p-value** was **≥ 0.05**:
*(Eğer görsel kontrol bir trend gösterdiyse veya ADF p-değeri ≥ 0.05 ise:)*

### 🛠️ Process (Süreç)

1.  **Compute the First Difference:**
    *(Birinci Farkı Hesapla:)*
    $$y'_t = y_t - y_{t-1}$$

2.  **Visual Check:**
    *(Görsel Kontrol:)*
    Does it now **oscillate around zero** with no clear drift?
    *(Şimdi belirgin bir sürüklenme olmadan sıfır etrafında salınıyor mu?)*

3.  **ADF Test Again:**
    *(Tekrar ADF Testi:)*
    If **p < 0.05**, accept `d=1`.
    *(Eğer p < 0.05 ise, d=1 olarak kabul et.)*

> ⚠️ **Note (Not):** Even if `d=0` passed, we sometimes try `d=1` to see if it improves **model stability**, but be careful not to **over-difference**.
> *(d=0 geçmiş olsa bile, bazen model kararlılığını iyileştirip iyileştirmediğini görmek için d=1 deneriz, ancak aşırı fark alma [over-differencing] konusunda dikkatli olun.)*


## 📌 Summary: Choosing `d`
*(Özet: d Seçimi)*

Aşağıdaki tablo, görsel analiz ve istatistiksel test sonuçlarına göre `d` parametresini nasıl seçeceğinizi özetler.

| Durum (Condition) | Aksiyon (Action) |
| :--- | :--- |
| **Visual:** Flat mean, no trend.<br>*(Düz ortalama, trend yok.)*<br>**ADF:** p < 0.05 | **Stop.** Accept `d=0`.<br>*(Dur. d=0 kabul et.)* |
| **Visual:** Trend visible.<br>*(Trend görünür.)*<br>**ADF:** p ≥ 0.05 | **Difference once.** Try `d=1`.<br>*(Bir kez fark al. d=1 dene.)* |
| **Visual:** Still trending after `d=1`.<br>*(d=1 sonrası hala trend var.)*<br>**ADF:** p ≥ 0.05 | **Difference again (Rare).** Try `d=2`.<br>*(Tekrar fark al [Nadir]. d=2 dene.)* |

---

> ⚠️ **Warning (Uyarı):** **Over-differencing** (*Aşırı Fark Alma*) can introduce extra **noise** (*gürültü*) and hurt your **forecast** (*tahmin*). Always balance **visual evidence** (*görsel kanıt*) with **statistical tests** (*istatistiksel testler*).


# 📉 Reading a PACF Plot: Choosing the AR Order (`p`)
*(PACF Grafiğini Okuma: AR Derecesi `p` Seçimi)*

ARIMA modelinin **AR (AutoRegressive)** bileşeni olan `p` parametresini belirlemek için birincil aracımız **Partial Autocorrelation Function (PACF)** grafiğidir. 

### 🧐 What is PACF?


ACF (Autocorrelation Function), bir gecikmenin (*lag*) hem doğrudan hem de dolaylı etkilerini gösterirken; **PACF**, aradaki gecikmelerin etkisini kaldırdıktan sonra, sadece o gecikmenin şimdiki zaman üzerindeki **saf ve doğrudan etkisini** (*pure/direct effect*) ölçer.

> 💡 **Rule of Thumb:** AR (`p`) derecesini bulmak için **PACF** grafiğine, MA (`q`) derecesini bulmak için **ACF** grafiğine bakılır.

---

### 📊 How to Interpret the PACF Plot

PACF grafiğindeki her bir çubuk (*bar*), ilgili gecikmenin korelasyon katsayısını temsil eder. Arka plandaki gölgeli alan (genellikle mavi), **95% Confidence Interval** (Güven Aralığı)'dır.


| Plot Feature (Grafik Özelliği) | Interpretation (Yorumlama) |
| :--- | :--- |
| **Tall bar outside the grey band**<br>*(Gri bandın dışındaki uzun çubuk)* | **Statistically Significant:** There is a significant, direct correlation at that lag.<br>*(İstatistiksel Olarak Anlamlı: O gecikmede anlamlı ve doğrudan bir etki vardır.)* |
| **Bars drop inside the band and stay there**<br>*(Çubuklar bandın içine düşüyor ve orada kalıyor)* | **Cut-off Point:** Useful memory ends here. This sharp drop indicates the order of the AR process.<br>*(Kesilme Noktası: Yararlı hafıza burada biter. Bu keskin düşüş AR sürecinin derecesini gösterir.)* |
| **First bar only**<br>*(Sadece birinci çubuk)* | **Classic AR(1):** Common in many time series. Only yesterday influences today.<br>*(Klasik AR(1): Birçok zaman serisinde yaygındır. Sadece dün bugünü etkiler.)* |
| **Several bars then sharp drop**<br>*(Birkaç çubuk sonra keskin düşüş)* | **AR(p):** Set `p` equal to the last significant lag before the drop.<br>*(AR(p): `p` değerini, düşüşten önceki son anlamlı gecikmeye eşitleyin.)* |

---

### 🛠️ Workflow: Choosing the AR Order `p`
*(İş Akışı: AR Derecesi `p` Seçimi)*

1.  **Stationarity Check:** Ensure the series is stationary. Difference (`d`) if needed.
    *(Durgunluk Kontrolü: Serinin durgun olduğundan emin olun. Gerekirse fark alın.)*
2.  **Plot PACF:** Use `plot_pacf(series, lags=30)` from `statsmodels`.
    *(PACF Çizimi: `statsmodels` kütüphanesini kullanın.)*
3.  **Identify Cut-off:** Find the **last bar** that sticks out significantly above the confidence interval.
    *(Kesilme Noktasını Belirle: Güven aralığının dışına çıkan son çubuğu bulun.)*
    * That lag number = **Candidate `p`** (*Aday p*).
4.  **Validate:** Don't rely solely on the plot. Compare models using **AIC/BIC** scores or **Cross-Validation**.
    *(Doğrulama: Sadece grafiğe güvenmeyin. AIC/BIC skorları veya Çapraz Doğrulama ile modelleri karşılaştırın.)*

> **🔍 Technical Detail:** The Confidence Interval is typically calculated as $\pm 1.96 / \sqrt{T}$ where $T$ is the number of observations. Bars within this range are considered **White Noise** (statistical noise).

---

### ❓ Common Questions & Troubleshooting

| Question | Quick Answer & Technical Reason  |
| :--- | :--- |
| **Why not pick a huge `p`?**<br>*(Neden çok büyük bir `p` seçmiyoruz?)* | **Overfitting Risk:** Adding too many lags captures random noise, not the signal. It increases model complexity without improving predictive power (penalized by AIC/BIC).<br>*(Aşırı Öğrenme Riski: Çok fazla gecikme sinyali değil gürültüyü yakalar. Tahmin gücünü artırmadan model karmaşıklığını yükseltir.)* |
| **What if I see no bars above the band?**<br>*(Ya bandın üzerinde hiç çubuk görmezsem?)* | **White Noise or Over-differencing:** The series might differenced too much (`d` is too high), or it holds no predictive pattern. Try `p=0` or reduce `d`.<br>*(Beyaz Gürültü veya Aşırı Fark Alma: Seri gereğinden fazla fark alınmış olabilir veya tahmin edilebilir bir desen içermiyordur.)* |
| **What if bars never drop (slow decay)?**<br>*(Ya çubuklar hiç düşmezse / yavaş azalırsa?)* | **Non-Stationarity:** The series is likely still non-stationary. Re-examine **differencing** (`d`) or check for **seasonality**.<br>*(Durgun Olmama: Seri muhtemelen hala durgun değildir. Fark alma işlemini tekrar inceleyin veya mevsimsellik kontrolü yapın.)* |

# 📉 Choosing the MA Order (`q`) with ACF
*(ACF ile MA Derecesi `q` Seçimi)*

ARIMA modelinin **MA (Moving Average - Hareketli Ortalama)** bileşeni olan `q` parametresini belirlemek için birincil aracımız **Autocorrelation Function (ACF)** grafiğidir.

### 🧐 What is `q` in ARIMA?
*(ARIMA'da `q` Nedir?)*

`p` parametresi geçmiş *değerlere* (satış rakamlarına) bakarken, **`q` parametresi geçmiş tahmin hatalarına (forecast errors/residuals)** bakar.
MA modelleri, serideki şokların veya hataların zaman içinde nasıl yayıldığını modeller.

> **Key Takeaway:** ACF answers "How many past mistakes still impact today?"
> *(Temel Çıkarım: ACF, "Geçmişteki kaç hata bugünü hala etkiliyor?" sorusuna cevap verir.)*

---

### 📊 How to Read an ACF Plot
*(ACF Grafiği Nasıl Okunur)*

MA (`q`) derecesini seçerken, ACF grafiğinde "Cut-off" (Kesilme) noktasına bakarız. PACF'in aksine, burada **ACF** grafiğindeki ani düşüşler MA derecesini işaret eder.



#### 🛠️ Step-by-Step Workflow (Adım Adım İş Akışı)

1.  **Stationarity First:** Ensure the series is stationary (`d` is fixed).
    *(Önce Durgunluk: Serinin durgun olduğundan emin olun.)*
2.  **Plot ACF:** Use `plot_acf(series)` from `statsmodels`.
    *(ACF Çizimi: `statsmodels` üzerinden ACF grafiğini çizin.)*
3.  **Identify Cut-off:** Look for the lag where bars drop into the **grey band** (Confidence Interval) and stay there.
    *(Kesilmeyi Belirle: Çubukların gri banda [Güven Aralığına] düştüğü ve orada kaldığı gecikmeyi bulun.)*
4.  **Set `q`:** The last significant lag before the drop is your candidate `q`.
    *(q'yu Ayarla: Düşüşten önceki son anlamlı gecikme, aday q değerinizdir.)*

---

### 🔍 Interpreting ACF Signatures
*(ACF İmzalarını Yorumlama)*

| Plot Pattern (Grafik Deseni) | Model Implication (Model Çıkarımı) |
| :--- | :--- |
| **Sharp Cut-off after Lag `q`**<br>*(Gecikme `q`'dan sonra keskin kesilme)* | **MA(`q`) Candidate:** Strong evidence for a Moving Average model of order `q`.<br>*(MA(q) Adayı: q derecesinden Hareketli Ortalama modeli için güçlü kanıt.)* |
| **Gradual Decay (Sine Wave / Exponential)**<br>*(Kademeli Azalma / Sinüs Dalgası)* | **AR(`p`) Process:** Typically indicates an Autoregressive process. Look at **PACF** instead to find `p`.<br>*(AR(p) Süreci: Genellikle Otokorelasyon sürecini gösterir. p'yi bulmak için PACF'e bakın.)* |
| **Spikes at Regular Intervals (s, 2s...)**<br>*(Düzenli Aralıklarda Sıçramalar)* | **Seasonality:** Indicates seasonal patterns (e.g., lag 7, 14, 21). Needs SARIMA.<br>*(Mevsimsellik: Mevsimsel kalıpları gösterir. SARIMA gerektirir.)* |

> 💡 **Technical Note:** For a pure MA(q) process:
> * **ACF:** Cuts off after lag `q`. (*Gecikme q'dan sonra kesilir.*)
> * **PACF:** Decays gradually (tails off). (*Kademeli olarak azalır.*)

---

### ⚠️ Common Pitfalls & Fixes
*(Yaygın Hatalar ve Çözümler)*

Model kurarken karşılaşabileceğiniz yaygın ACF desenleri ve teknik çözümleri:

| Pitfall (Tuzak) | Symptom on ACF (ACF Belirtisi) | Technical Fix (Teknik Çözüm) |
| :--- | :--- | :--- |
| **Over-differencing**<br>*(Aşırı Fark Alma)* | **No significant bars:** First lag might even be significantly negative (approx -0.5).<br>*(Anlamlı çubuk yok: İlk gecikme negatif ve anlamsız olabilir.)* | **Try smaller `d`:** You likely differenced a stationary series unnecessarily. Revert to `d=0` or `d-1`.<br>*(Daha küçük `d` dene: Muhtemelen durgun bir serinin gereksiz yere farkını aldınız.)* |
| **Under-differencing**<br>*(Yetersiz Fark Alma)* | **Very slow decay:** Bars stay high and positive for many lags, decreasing linearly.<br>*(Çok yavaş azalma: Çubuklar birçok gecikme boyunca yüksek ve pozitif kalır, doğrusal azalır.)* | **Increase `d`:** The series is still non-stationary (Unit Root present). Take an additional difference.<br>*(d'yi Artır: Seri hala durgun değil. Bir fark daha alın.)* |
| **Seasonality Present**<br>*(Mevsimsellik Var)* | **Periodic Spikes:** Significant correlations appearing at specific lags (e.g., 7, 14 for weekly data).<br>*(Periyodik Sıçramalar: Belirli gecikmelerde [haftalık veride 7, 14 gibi] anlamlı korelasyonlar.)* | **Seasonal Model:** Consider **SARIMA** (adding Seasonal MA term `Q`) or apply **Seasonal Differencing** (`D=1`, lag=7).<br>*(Mevsimsel Model: SARIMA düşünün veya Mevsimsel Fark Alma uygulayın.)* |

---

### ✅ Summary Strategy: Choosing `p` and `q`
*(Özet Strateji: p ve q Seçimi)*

| Parameter | Plot to Watch | Pattern to Look For |
| :--- | :--- | :--- |
| **AR (`p`)** | **PACF** | **Cut-off:** Last significant spike determines `p`. |
| **MA (`q`)** | **ACF** | **Cut-off:** Last significant spike determines `q`. |

> **Final Check:** After choosing `p` and `q`, always validate with **Information Criteria (AIC/BIC)** and check the residuals of your model.
> *(Son Kontrol: Seçimden sonra her zaman Bilgi Kriterleri [AIC/BIC] ile doğrulayın ve model hatalarını [residuals] kontrol edin.)*


# 📌 Summary & ARIMA Workflow
*(Özet ve ARIMA İş Akışı)*

### 🚀 Why ARIMA?
*(Neden ARIMA?)*

**ARIMA** (*AutoRegressive Integrated Moving Average*), daha ağır makine öğrenimi yaklaşımlarına (*Machine Learning approaches*) dalmadan önce başvurmanız gereken, **kompakt**, **şeffaf** ve **istatistiksel olarak titiz** bir tahmin modelidir.

> **⚠️ Limitation (Kısıt):**
> ARIMA, by design, handles **trend** and short-term **autocorrelation** but assumes your series has **no built-in seasonality**. There is no term in the definition explicitly modeling repeating cycles.
> *(ARIMA, tasarımı gereği trendi ve kısa vadeli otokorelasyonu yönetir ancak serinizin yerleşik bir mevsimselliği olmadığını varsayar. Tanımında tekrarlayan döngüleri modelleyen açık bir terim yoktur.)*
>
> 💡 **Solution:** If your data shows strong weekly or annual patterns, upgrade to **SARIMA**, which adds a "Seasonal" component.

---

### ⚙️ The Three Pillars: p, d, q
*(Üç Temel Direk: p, d, q)*

ARIMA requires three hyperparameters that define its structure:

| Parameter | Component | Description (Açıklama) |
| :--- | :--- | :--- |
| **p** | **AutoRegression (AR)**<br>*(Oto-Regresyon)* | **Past Values:** It looks back at the last `p` days of sales to detect patterns.<br>*(Geçmiş Değerler: Kalıpları tespit etmek için son p günün satışlarına bakar.)* |
| **d** | **Integration (I)**<br>*(Entegrasyon/Fark Alma)* | **Stationarity:** It removes smooth up-or-down trends by **differencing** the data `d` times.<br>*(Durgunluk: Verinin d kez farkını alarak yukarı veya aşağı yönlü trendleri kaldırır.)* |
| **q** | **Moving Average (MA)**<br>*(Hareketli Ortalama)* | **Past Errors:** It learns from the last `q` days of **forecast errors** to correct its predictions.<br>*(Geçmiş Hatalar: Tahminlerini düzeltmek için son q günün tahmin hatalarından öğrenir.)* |

---

### 🛠️ The Professional Workflow
*(Profesyonel İş Akışı)*

Follow this step-by-step pipeline to build a robust ARIMA model.

#### 1. Exploratory Analysis
*(Keşifçi Analiz)*
* **Visualize:** Plot the raw series, moving average, and variance.
* **Check Trend:** Is there an obvious upward/downward drift?
* **Test Stationarity:** Run the **ADF Test** (*Augmented Dickey-Fuller*).

#### 2. Differencing (Parameter `d`)
*(Fark Alma)*
* Difference the series (`d` times) until it looks flat (constant mean).
* **Verification:** Ensure ADF p-value < 0.05.
* *Note:* Usually `d=1` is sufficient. `d=2` is rare.

#### 3. Identification (Parameters `p` & `q`)
*(Tanımlama)*
Plot **ACF** and **PACF** charts on the **differenced** series:
* **PACF Cut-off:** Suggests the AR order (**p**).
* **ACF Cut-off:** Suggests the MA order (**q**).

#### 4. Model Fitting
*(Model Eğitimi)*
Fit the model using `statsmodels`.

```python
from statsmodels.tsa.arima.model import ARIMA

# Define the model with identified orders
# order = (p, d, q)
model = ARIMA(series, order=(p, d, q))

# Train the model
results = model.fit()

# View statistical summary
print(results.summary())

```

### 5. 🩺 Diagnostics: Residual Analysis
*(Tanılama: Artık Analizi)*

This is a **critically important step** (*kritik derecede önemli bir adım*). We must check if the **residuals** (*hatalar/artıklar*) resemble **White Noise** (*Beyaz Gürültü*).

**Checklist for Residuals:**
*(Artıklar için Kontrol Listesi:)*

* **Mean** (*Ortalama*): Should be close to **0**.
    *(0'a yakın olmalıdır.)*
* **Variance** (*Varyans*): Should be **constant**.
    *(Sabit olmalıdır.)*
* **No Correlation** (*Korelasyon Yok*): The **ACF** of residuals should show no significant **spikes**.
    *(Artıkların ACF grafiği anlamlı sivrilmeler göstermemelidir.)*
* **Test:** Use the **Ljung-Box Test** to statistically confirm residuals are **random**.
    *(Test: Artıkların rastgele olduğunu istatistiksel olarak doğrulamak için Ljung-Box Testi kullanın.)*

---

### 6. 🔮 Forecast and Evaluate
*(Tahmin ve Değerlendirme)*

Once the diagnostics pass, we evaluate the model using specific metrics.

#### 📉 Model Selection Metric: AIC
*(Model Seçim Metriği: AIC)*

* **Definition:** **AIC (Akaike Information Criterion)**.
* **Purpose:** Used for **Model Selection**.
    *(Model Seçimi için kullanılır.)*
* **Interpretation:** **Lower is better**. It balances **model fit** vs. **complexity**.
    *(Daha düşük olması daha iyidir. Model uyumu ile karmaşıklığı dengeler.)*

#### 🎯 Accuracy Metrics: MAE / RMSE
*(Doğruluk Metrikleri: MAE / RMSE)*

* **Purpose:** Used for **Accuracy**.
    *(Doğruluk için kullanılır.)*
* **Interpretation:** Evaluate how close the **predictions** are to **actuals** on a test set.
    *(Test setindeki tahminlerin gerçek değerlere ne kadar yakın olduğunu ölçer.)*

