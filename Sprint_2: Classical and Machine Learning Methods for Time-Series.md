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

# 🗓️ Classical Time-Series Methods: SARIMA
*(Klasik Zaman Serisi Yöntemleri: SARIMA)*

**SARIMA** (*Seasonal AutoRegressive Integrated Moving Average*), klasik ARIMA modelinin, verilerdeki **mevsimsel döngüleri** (*seasonal cycles*) modelleyebilecek şekilde genişletilmiş halidir.

### 🚀 Why "ARIMA with an S" is the Next Logical Step?
*(Neden "S" eklenmiş ARIMA bir sonraki mantıksal adımdır?)*

Standard ARIMA models capture a series’ **short-term memory** (*kısa vadeli hafıza*). However, real-world data (e.g., supermarket sales, airline passengers) often repeats patterns every week, month, or quarter.
*(Standart ARIMA modelleri serinin kısa vadeli hafızasını yakalar. Ancak gerçek dünya verileri genellikle her hafta, ay veya çeyrekte tekrarlayan desenler içerir.)*

> **The Problem:** Traditional ARIMA can’t model repeating boosts (like a Saturday spike) unless you manually inject complex lags.
> **The Solution:** SARIMA adds a **second, parallel ARIMA layer** that only "wakes up" at the seasonal interval $s$.

---

### 📊 Comparative Analysis Matrix: SARIMA Architecture
*(Karşılaştırmalı Analiz Matrisi: SARIMA Mimarisi)*

Bu tablo, SARIMA'nın bileşenlerini, çözdüğü problemleri ve teknik detaylarını analiz eder.

| Analysis Area (Analiz Alanı) | Problems & Components (Sorunlar ve Bileşenler) | Technical Detail & Importance (Teknik Detay ve Önem) | Solution Methods (Çözüm Yöntemleri) | Tools & Tests (Araçlar ve Testler) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Seasonality Handling**<br>*(Mevsimsellik Yönetimi)* | **Problem:** ARIMA'nın mevsimsel şokları (örn. Noel satışları) görememesi.<br>**Component:** **Seasonal Period ($s$)**. | **Detail:** $s$, döngünün uzunluğudur.<br>• Haftalık veri için $s=7$.<br>• Aylık veri için $s=12$.<br>**Importance:** Modelin hangi aralıklarla geçmişe bakacağını belirler. | **Notation:**<br>$$SARIMA(p, d, q) \times (P, D, Q)_s$$<br>Mevsimsel olmayan ve mevsimsel parametrelerin çarpımı. | • **Seasonal Decomposition:** Trend ve mevsimselliği görsel ayırma.<br>• **ACF Plot:** $s, 2s, 3s$ gecikmelerindeki sıçramaları kontrol etme. |
| **2. Seasonal AutoRegression ($P$)**<br>*(Mevsimsel Oto-Regresyon)* | **Problem:** Bu ayın satışlarının, geçen yılın aynı ayındaki satışlarla ilişkisi.<br>**Component:** **Seasonal AR ($P$)**. | **Detail:** "Kaç tane mevsimsel dün ($t-s, t-2s$) bugünü etkiliyor?" sorusuna yanıt verir.<br>**Importance:** Geçmiş sezonların momentumunu bugüne taşır. | **Interaction:**<br>Standart $p$ (dün) ile Mevsimsel $P$ (geçen yıl bugün) birlikte çalışır. | • **PACF Plot:** $s$ katlarında (12, 24...) keskin düşüşler aranır.<br>• **Grid Search:** En iyi $P$ değerini deneme yanılma ile bulma. |
| **3. Seasonal Differencing ($D$)**<br>*(Mevsimsel Fark Alma)* | **Problem:** Mevsimsel trendler (yıldan yıla artan yaz trafiği).<br>**Component:** **Seasonal Integrated ($D$)**. | **Detail:** Mevsimsel seviye kaymalarını (*Level Shifts*) kaldırmak için fark alır.<br>Formül: $y_t - y_{t-s}$.<br>**Importance:** Veriyi mevsimsel olarak durgunlaştırır (*Seasonally Stationary*). | **Method:**<br>Genellikle $D=1$ yeterlidir. Bu, seriden "geçen yılın aynı ayını çıkarma" işlemidir. | • **Canova-Hansen Test:** Mevsimsel kararlılık testi.<br>• **Visual Check:** $s$ periyodunda tekrar eden dalgaların düzleşmesi. |
| **4. Seasonal Moving Average ($Q$)**<br>*(Mevsimsel Hareketli Ortalama)* | **Problem:** Geçmiş sezonlardaki tahmin hatalarının bugüne etkisi.<br>**Component:** **Seasonal MA ($Q$)**. | **Detail:** "Kaç tane mevsimsel hata şoku (*Error Shocks*) kalıcı oluyor?"<br>Örn: Geçen Aralık'taki tahmin hatası bu Aralık'ı düzeltir.<br>**Importance:** Tahminlerin mevsimsel sapmalara karşı dirençli olmasını sağlar. | **Calculation:**<br>Model, $t-s$ zamanındaki hatayı ($e_{t-s}$) kullanarak bugünkü tahmini revize eder. | • **ACF Plot:** $s$ gecikmesindeki (lag $s$) negatif korelasyon veya kesilme noktası. |
| **5. Model Tuning & Selection**<br>*(Model Ayarlama ve Seçim)* | **Problem:** Toplam 7 parametrenin ($p,d,q,P,D,Q,s$) optimizasyonu.<br>**Component:** **Hyperparameter Tuning**. | **Detail:** Çok sayıda kombinasyon olduğu için model karmaşıklığı artar.<br>**Importance:** Yanlış $s$ veya $D$ seçimi, tamamen hatalı tahminlere yol açar. | **Auto-ARIMA:**<br>`pmdarima` gibi kütüphanelerle parametrelerin otomatik denenmesi (AIC minimizasyonu). | • **AIC/BIC:** Model karşılaştırma.<br>• **Ljung-Box:** Mevsimsel hataların rastgeleliğini test etme. |

---

### 🧩 The Structure of SARIMA
*(SARIMA'nın Yapısı)*

SARIMA model is denoted as:
*(SARIMA modeli şu şekilde gösterilir:)*

$$SARIMA(p, d, q) \times (P, D, Q)_s$$



#### 1. Non-Seasonal Part $(p, d, q)$
*(Mevsimsel Olmayan Kısım)*
Standart ARIMA parametreleri. Günlük/kısa vadeli trendleri ve otokorelasyonu yakalar.

#### 2. Seasonal Part $(P, D, Q)_s$
*(Mevsimsel Kısım)*
* **P (Seasonal AR):** Looks at past seasonal values.
    *(Geçmiş mevsimsel değerlere bakar. Örn: Geçen yılın aynı ayı.)*
* **D (Seasonal Differencing):** Removes repeating seasonal patterns.
    *(Tekrarlayan mevsimsel desenleri kaldırır. $Series.diff(s)$ işlemi.)*
* **Q (Seasonal MA):** Looks at past seasonal errors.
    *(Geçmiş mevsimsel hatalara bakar.)*
* **s (Period):** The length of the cycle.
    *(Döngünün uzunluğu. Haftalık için 7, Yıllık (aylık veri) için 12.)*

---

### 💡 Key Takeaway
*(Temel Çıkarım)*

**SARIMA is like ARIMA, but it’s designed for data that repeats in cycles.**
*(SARIMA, ARIMA gibidir ancak döngüler halinde tekrarlayan [haftalık veya yıllık satış kalıpları gibi] veriler için tasarlanmıştır.)*

While standard ARIMA handles general trends, **SARIMA** is indispensable when "this December" depends heavily on "last December".
*(Standart ARIMA genel trendleri yönetirken, "bu Aralık" ayının büyük ölçüde "geçen Aralık" ayına bağlı olduğu durumlarda SARIMA vazgeçilmezdir.)*

# 🗓️ Choosing Seasonal Orders for SARIMA: Step 1
*(SARIMA İçin Mevsimsel Derecelerin Seçimi: Adım 1)*

Today’s lecture walks you through picking the right **seasonal orders** ($P, D, Q$) and combining them with your **non-seasonal** ($p, d, q$) parameters.
*(Bugünkü ders, doğru mevsimsel dereceleri [P, D, Q] seçmenize ve bunları mevsimsel olmayan [p, d, q] parametrelerinizle birleştirmenize rehberlik eder.)*

### 🗺️ Our Roadmap
*(Yol Haritamız)*

1.  **Confirm the seasonal period $s$.**
    *(Mevsimsel periyodu s'yi doğrula.)*
2.  **Stationarize** with ordinary and seasonal **differencing**.
    *(Normal ve mevsimsel fark alma ile durgunlaştır.)*
3.  Read **seasonal ACF/PACF spikes** to guess $P$ and $Q$.
    *(P ve Q'yu tahmin etmek için mevsimsel ACF/PACF sıçramalarını oku.)*
4.  **Grid-search** a small set of $(p, q) \times (P, Q)$.
    *(Küçük bir (p, q) x (P, Q) seti üzerinde ızgara araması yap.)*
5.  **Fit the models** to the grid.
    *(Modelleri ızgaraya uydur/eğit.)*
6.  **Select the best model** by **AIC/BIC** and **hold-out error**.
    *(AIC/BIC ve dışarıda tutulan set hatasına göre en iyi modeli seç.)*

---

## 🕵️ Step 1: Confirm the Seasonal Period
*(Adım 1: Mevsimsel Periyodu Doğrulama)*

Before you can set any seasonal parameters, you must know **how long the season is**. This is the **single most important input** to SARIMA.
*(Herhangi bir mevsimsel parametre ayarlamadan önce, sezonun ne kadar sürdüğünü bilmelisiniz. Bu, SARIMA için en önemli tek girdidir.)*

### 1. Domain Knowledge (Fast Check)
*(Alan Bilgisi - Hızlı Kontrol)*

* Does the business talk about "**weekly cycles**" (*haftalık döngüler*), "**monthly billing peaks**" (*aylık fatura zirveleri*), or "**quarterly budgets**" (*çeyreklik bütçeler*)?
* **Daily Data:** If operations close every weekend, start with **$s = 7$**.
* **Hourly Data:** Call-center data often cycle every 24 hours → **$s = 24$**.

### 2. Visual Inspection
*(Görsel İnceleme)*

We will work with the same dataframe as we did with ARIMA.
Plot the raw series: `train.plot()`
Look for regularly spaced **ridges** (*sırtlar/tepeler*) or **troughs** (*çukurlar*). Count the spacing in days, hours, or months—that spacing is your candidate $s$.



> **💡 Think First!**
> What do you see in the plot?
> *(Grafikte ne görüyorsunuz?)*

**📉 Our Analysis:**
* From a quick visual scan, you can see "**ridges**" (higher-than-average clusters of points) and "**troughs**" repeating at a fairly even **cadence** (*ritim/ahenk*).
* If you mark any spike and count forward to the next one of similar height, you hit roughly the same spot after about **seven days** each time.
* **Result:** That makes **7 days** the most plausible seasonal period for this daily series—a classic **weekly cycle**.

### 3. Inspect the ACF for Spikes
*(Sıçramalar İçin ACF'yi İnceleme)*

Plot the plain **ACF** up to, say, $3 \times$ your suspected period.
*(Normal ACF'yi, şüphelendiğiniz periyodun 3 katına kadar çizdirin.)*

```python
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(train.values.flatten(), lags=30)  # Look for bars at s, 2s, 3s...
```

### 📊 Analyzing the ACF Plot for Seasonality
*(Mevsimsellik İçin ACF Grafiğini Analiz Etme)*

ACF grafiği, mevsimsel periyodu ($s$) doğrulamak için en güçlü kanıtı sunar.

#### 🚦 Signals (Sinyaller)

* **Positive Signal** (*Pozitif Sinyal*): If vertical bars appear at lag $s, 2s, 3s \dots$ while the in-between lags are small, you’ve found a **seasonal pulse** (*mevsimsel nabız*) at $s$.
    *(Dikey çubuklar s, 2s, 3s... gecikmelerinde beliriyor ve aradaki gecikmeler küçük kalıyorsa, s noktasında mevsimsel bir etki bulmuşsunuz demektir.)*
* **Negative Signal** (*Negatif Sinyal*): If no spikes appear, a **seasonal model** may not help.
    *(Herhangi bir sıçrama görünmüyorsa, mevsimsel bir model yardımcı olmayabilir.)*



---

### 💡 Think First!


> **Reflection:** What do you see in the plot?
> *(Grafikte ne görüyorsunuz?)*

---

### 📉 Our Analysis


Based on the ACF plot, here is the technical breakdown:
*(ACF grafiğine dayanarak, işte teknik döküm:)*

#### 1. Strong Positive Lag 1 Spike
*(Güçlü Pozitif 1. Gecikme Sıçraması)*
* **Observation:** The correlation at Lag 1 is very high.
* **Interpretation:** Yesterday’s sales are a very good predictor of today’s—this represents classic **short-memory momentum**.
    *(Dünkü satışlar bugünün çok iyi bir tahmincisidir—bu, klasik kısa vadeli momentumu/hafızayı temsil eder.)*

#### 2. Regular Positive Spikes at Lags 7, 14, 21, 28
*(7, 14, 21, 28. Gecikmelerde Düzenli Pozitif Sıçramalar)*
* **Observation:** Every seventh day, the autocorrelation climbs back to roughly **0.5**.
* **Interpretation:** This confirms a clear **weekly cycle** (*haftalık döngü*). It validates the "**same-day-last-week**" (*geçen hafta aynı gün*) effect.

#### 3. Alternating Negative Bars
*(Ardışık Negatif Çubuklar)*
* **Observation:** Values about half a week apart move in **opposite directions**.
* **Interpretation:** This is typical for a series with **steady seasonality** (*istikrarlı mevsimsellik*) but no long-term trend.
    *(Yaklaşık yarım hafta aralıklı değerler zıt yönlerde hareket ediyor; bu, istikrarlı mevsimselliği olan ancak uzun vadeli trendi olmayan bir seri için tipiktir.)*

---

### ✅ Conclusion for Step 1
*(Adım 1 İçin Sonuç)*

Based on the Domain Knowledge, Visual Inspection, and ACF Analysis:
*(Alan Bilgisi, Görsel İnceleme ve ACF Analizine dayanarak:)*

* **Season length ($m$ or $s$) = 7.**
* The **weekly pattern** (*haftalık desen*) is dominant.
    *(Sezon uzunluğu [m veya s] = 7. Haftalık desen baskın.)*

We will proceed with **$s=7$** for our SARIMA model.

# 🗓️ SARIMA Modelling: Steps 2 & 3
*(SARIMA Modelleme: Adım 2 ve 3)*

Mevsimsel periyodu ($s=7$) doğruladıktan sonra, seriyi **durgunlaştırmalı** (*stationarize*) ve mevsimsel parametreleri ($P, Q$) belirlemeliyiz.

---

## 📉 Step 2: Seasonal Differencing ($D$)
*(Adım 2: Mevsimsel Fark Alma)*

To apply SARIMA effectively, we need to handle two types of trends:
*(SARIMA'yı etkili bir şekilde uygulamak için iki tür trendi yönetmemiz gerekir:)*

1.  **Ordinary Differencing ($d$):** Wipes out the **long-term trend**.
    *(Sıradan Fark Alma: Uzun vadeli trendi siler.)*
2.  **Seasonal Differencing ($D$):** Flattens the **repeating level** every $s$ periods.
    *(Mevsimsel Fark Alma: Her s periyodunda tekrarlayan seviyeyi düzleştirir.)*

> **Goal:** After differencing, the **ACF** should no longer show a slow, stair-step decay, and the **mean/variance** should look stable.
> *(Hedef: Fark alma işleminden sonra, ACF artık yavaş, basamaklı bir azalma göstermemeli ve ortalama/varyans kararlı görünmelidir.)*

---

## 🕵️ Step 3: Guess ($P, Q$) from Seasonal Lags
*(Adım 3: Mevsimsel Gecikmelerden P ve Q Tahmini)*

We read the **seasonal lags** in ACF & PACF to guess the orders.
*(Dereceleri tahmin etmek için ACF ve PACF'teki mevsimsel gecikmeleri okuruz.)*

### 🛠️ The Process (Süreç)
1.  **Run PACF/ACF** on the **seasonally differenced series** (`diff_season`).
    *(Mevsimsel farkı alınmış seri üzerinde PACF/ACF çalıştırın.)*
2.  **Look at bars at lags $s, 2s, 3s \dots$** (e.g., 7, 14, 21).
    *(s, 2s, 3s... gecikmelerindeki çubuklara bakın.)*

### 📏 Rules of Thumb (Pratik Kurallar)

| Plot | Pattern at Lag $s$ | Initial Guess |
| :--- | :--- | :--- |
| **Seasonal PACF** | Big spike at lag $s$ (*Lag s'de büyük sıçrama*) | Start with **$P = 1$** (Seasonal AR) |
| **Seasonal ACF** | Big spike at lag $s$ (*Lag s'de büyük sıçrama*) | Start with **$Q = 1$** (Seasonal MA) |

> **Note:** If spikes persist at $2 \times s$, test $P=2$ or $Q=2$; if they vanish after the first spike, one term is usually enough.

---



## 📉 Our Analysis: Interpreting the Plots
*(Analizimiz: Grafikleri Yorumlama)*

We applied **Seasonal Differencing** ($D=1, m=7$). Here is what the plots tell us now:
*(Mevsimsel Fark Alma uyguladık. İşte grafiklerin bize söyledikleri:)*

<img width="441" height="534" alt="image" src="https://github.com/user-attachments/assets/795c4ccc-6d30-4cab-913a-717efee62867" />


### 1. ACF Analysis (Top Chart)
* **Seasonal Pattern Gone:** No more big spikes at lags 7, 14, 21… This means the **weekly seasonality is gone**.
    *(Mevsimsel Desen Gitti: 7, 14, 21... gecikmelerinde artık büyük sıçramalar yok. Haftalık mevsimsellik ortadan kalktı.)*
* **Success:** Seasonal differencing ($D=1, m=7$) worked; the series is now **seasonally stationary**.
    *(Başarı: Mevsimsel fark alma işe yaradı; seri artık mevsimsel olarak durgun.)*
* **Non-Seasonal Hint:** The strong **lag-1 negative autocorrelation** suggests we should keep a short non-seasonal MA term (**$q=1$**).
    *(Mevsimsel Olmayan İpucu: Güçlü negatif lag-1 otokorelasyonu, kısa bir mevsimsel olmayan MA terimi tutmamız gerektiğini önerir.)*

### 2. PACF Analysis (Bottom Chart)
* **Non-Seasonal AR:** Significant negative partial autocorrelations at lags 1 to 5, then everything dies out.
    *(Mevsimsel Olmayan AR: 1'den 5'e kadar olan gecikmelerde anlamlı negatif kısmi otokorelasyonlar var, sonra sönümleniyor.)*
* **Implication:** A finite AR structure of about **$p \approx 5$** is enough to explain the remaining correlation. This matches our previous ARIMA choice ($p=6$).
    *(Çıkarım: Yaklaşık p=5 olan sonlu bir AR yapısı, kalan korelasyonu açıklamak için yeterlidir.)*

### 3. Choosing $P$ and $Q$ (The Decision)
* **Observation:** After seasonal differencing, you’d expect remaining seasonality to show up as spikes at lags 7, 14, 21.
* **Reality:** We have **somehow still significant small spikes**, so it seems the weekly cycle has kind of been removed but traces remain.
    *(Gerçeklik: Hala bir şekilde anlamlı küçük sıçramalarımız var, bu da haftalık döngünün kısmen kaldırıldığını ancak izlerin kaldığını gösteriyor.)*

> **✅ Final Decision:** Therefore, we will establish **$P = 1$** and **$Q = 1$**.
> *(Son Karar: Bu nedenle, P=1 ve Q=1 olarak belirleyeceğiz.)*


# 🗓️ SARIMA Modelling: Step 4 - Combine & Grid Search
*(SARIMA Modelleme: Adım 4 - Birleştirme ve Izgara Araması)*

Mevsimsel ($P, D, Q$) ve Mevsimsel Olmayan ($p, d, q$) parametreler için ilk tahminlerimizi yaptık. Şimdi, en iyi kombinasyonu bulmak için sistematik bir arama yapacağız.

---

## 🕸️ The Strategy: Grid-Search a Small Neighbourhood
*(Strateji: Küçük Bir Komşulukta Izgara Araması)*

Instead of testing every possible number (which takes forever), we define a **small neighbourhood** (*küçük bir komşuluk/aralık*) around our initial guesses.
*(Her olası sayıyı test etmek yerine [ki bu sonsuza kadar sürer], ilk tahminlerimizin etrafında küçük bir aralık tanımlıyoruz.)*

### 🛠️ The Rules (Kurallar)
1.  **Keep `d` and `D` fixed:** We already confirmed stationarity.
    *(d ve D'yi sabit tutun: Durgunluğu zaten doğruladık.)*
2.  **Try a short list of non-seasonal orders:** e.g., $(p, q) = (1,1), (2,1)$.
    *(Kısa bir mevsimsel olmayan dereceler listesi deneyin.)*
3.  **Combine with seasonal pairs:** e.g., $(P, Q) = (1,1)$ or $(0,1)$.
    *(Mevsimsel çiftlerle birleştirin.)*

> **🚀 Efficiency Tip:** A handful of runs is usually enough to find the **sweet spot** (*en uygun nokta*) without **over-computing** (*aşırı hesaplama/işlem yükü*).

---

## 📊 Designing the Grid
*(Izgarayı Tasarlama)*

Based on our previous ACF/PACF analysis, here is our search space:

| Parameter | Initial Guess (*İlk Tahmin*) | Neighbourhood to Try (*Denenen Aralık*) | Reasoning (*Mantık*) |
| :--- | :--- | :--- | :--- |
| **p** | 5 | **4 – 6** | PACF showed significant lags up to ~5. (*PACF 5'e kadar anlamlıydı.*) |
| **q** | 1 | **0 – 1** | ACF lag-1 was strong. (*ACF lag-1 güçlüydü.*) |
| **P** | 1 | **0 – 1** | Only if seasonal PACF at lag 7 looks non-zero. (*Sadece lag 7'de PACF sıfır değilse.*) |
| **Q** | 1 | **0 – 1** | If seasonal ACF at lag 7 resurfaces. (*Eğer lag 7'de ACF tekrar belirirse.*) |
| **D** | 1 | **Fixed (1)** | Seasonal differencing already helped. (*Mevsimsel fark alma zaten işe yaradı.*) |

---

## 💻 The Resulting Grid Code
*(Ortaya Çıkan Izgara Kodu)*

We will create two lists of tuples to iterate through.
*(Üzerinde döngü kuracağımız iki demet listesi oluşturacağız.)*

```python
# 1. Non-Seasonal Combinations (p, d, q)
# We test p around 5, keeping d=0 (from Step 1 stationarity), and q around 1.
pdq_combinations = [
    (4, 0, 0), (4, 0, 1),
    (5, 0, 0), (5, 0, 1),
    (6, 0, 0), (6, 0, 1)
]

# 2. Seasonal Combinations (P, D, Q, s)
# We test P and Q around 0-1, keeping D=1 and s=7.
seasonal_combinations = [
    (0, 1, 0, 7),
    (0, 1, 1, 7),
    (1, 1, 0, 7),
    (1, 1, 1, 7)
]
```

# 🗓️ SARIMA Modelling: Step 5 - Forecast & Implementation
*(SARIMA Modelleme: Adım 5 - Tahmin ve Uygulama)*

Model parametrelerini belirledikten sonra, tahmin üretme ve performansı değerlendirme aşamasına geçiyoruz. Ayrıca, bu bölümde **DARTS** kütüphanesinde SARIMA'nın nasıl tanımlandığını göreceğiz.

---

## 🔮 Forecasting Strategy
*(Tahmin Stratejisi)*

Once the model is fitted, we evaluate its performance using standard metrics.
*(Model eğitildikten sonra, performansını standart metrikler kullanarak değerlendiririz.)*

1.  **Generate Forecasts:** Predict future values for the validation period.
    *(Gelecek tahminleri üretin: Doğrulama periyodu için gelecekteki değerleri tahmin edin.)*
2.  **Compare Against Baseline:** Compare **MAE** (*Mean Absolute Error*) and **RMSE** (*Root Mean Squared Error*) against a **Naïve Seasonal Baseline**.
    *(Temel Referansla Karşılaştırın: MAE ve RMSE'yi Saf Mevsimsel Referans ile karşılaştırın.)*
    > **Naïve Seasonal Baseline:** Repeats the value from $t-s$ (e.g., predicting next Monday's sales as exactly last Monday's sales).

---

## 🛠️ Implementation in DARTS
*(DARTS ile Uygulama)*

To model seasonality with ARIMA in DARTS, we utilize the `seasonal_order` parameter. This effectively mimics the SARIMA functionality found in `statsmodels`.

### The `seasonal_order` Parameter
*(seasonal_order Parametresi)*

We define the tuple `(P, D, Q, m)`:

* **P:** Seasonal Autoregressive order (*Mevsimsel AR derecesi*)
* **D:** Seasonal Differencing order (*Mevsimsel Fark Alma derecesi*)
* **Q:** Seasonal Moving Average order (*Mevsimsel MA derecesi*)
* **m:** Periodicity of the seasonal component (*Mevsimsel bileşenin periyodu*)
    * *Example:* $m=7$ for weekly seasonality in daily data.

### 💻 Code Example
*(Kod Örneği)*

We will configure and fit a SARIMA model using the specific seasonal order identified in our analysis `(1, 1, 1, 7)` combined with a standard ARIMA order (e.g., `p=5, d=0, q=1`).

```python
from darts.models import ARIMA

# 1. Initialize the Model
# order=(p, d, q) -> Non-seasonal parameters
# seasonal_order=(P, D, Q, m) -> Seasonal parameters
model_sarima = ARIMA(
    order=(5, 0, 1),           # From Grid Search Step 4
    seasonal_order=(1, 1, 1, 7) # P=1, D=1, Q=1, m=7
)

# 2. Fit the Model
# Assuming 'train' is your Darts TimeSeries object
model_sarima.fit(train)

# 3. Forecast
# Predict the next n steps (e.g., 30 days)
forecast = model_sarima.predict(n=30)

# 4. Visualization (Optional)
# train.plot(label="Train")
# forecast.plot(label="Forecast")

```

* After training the model on the training data, we forecast and plot the results:

* <img width="652" height="235" alt="image" src="https://github.com/user-attachments/assets/ce17741e-4dda-4f75-86d9-22d976ba966c" />

# 🏆 Final Step: Evaluation & Success Criteria
*(Son Adım: Değerlendirme ve Başarı Kriterleri)*

Bir model eğitmek işin sadece yarısıdır. Diğer yarısı ise şu soruyu yanıtlamaktır: **"Bu model, basit bir tahminden gerçekten daha mı iyi?"**

---

## 📏 The Benchmark: Naïve Seasonal Baseline
*(Kıyaslama Noktası: Saf Mevsimsel Referans)*

Before celebrating low error rates, we must compare our complex SARIMA model against a "dumb" but effective baseline.
*(Düşük hata oranlarını kutlamadan önce, karmaşık SARIMA modelimizi "basit" ama etkili bir referans noktasıyla kıyaslamalıyız.)*

### 🧐 What is Naïve Seasonal Baseline?
*(Saf Mevsimsel Referans Nedir?)*
It simply repeats the value from the last season ($t-s$).
*(Basitçe son sezonun [t-s] değerini tekrar eder.)*

* **Logic:** "Next Monday's sales will be exactly the same as last Monday's sales."
    *(Mantık: "Gelecek Pazartesi'nin satışları, geçen Pazartesi'nin satışlarıyla birebir aynı olacak.")*
* **Formula:** $\hat{y}_t = y_{t-s}$

> **Why do we need this?** If your complex SARIMA model cannot beat this simple logic, then the model is **over-engineered** and useless.
> *(Neden buna ihtiyacımız var? Eğer karmaşık SARIMA modeliniz bu basit mantığı geçemiyorsa, model aşırı mühendislik ürünüdür ve yararsızdır.)*

---

## 📊 Performance Comparison
*(Performans Karşılaştırması)*

We evaluate success by comparing error metrics (**MAE** or **RMSE**) between the SARIMA Forecast and the Baseline.



### 📉 Metrics to Watch (İzlenecek Metrikler)

| Metrik (Metric) | Baseline Score (Örn.) | SARIMA Score (Örn.) | Interpretation (Yorum) |
| :--- | :--- | :--- | :--- |
| **MAE** | 50.5 | **32.1** | **Significant Improvement:** SARIMA reduced the average error by ~36%.<br>*(Anlamlı İyileşme: SARIMA ortalama hatayı ~%36 azalttı.)* |
| **RMSE** | 65.2 | **41.8** | **Better Stability:** The model handles large outliers better than the baseline.<br>*(Daha İyi Kararlılık: Model, büyük aykırı değerleri referanstan daha iyi yönetiyor.)* |

---

## ✅ Success Criteria Checklist
*(Başarı Kriterleri Kontrol Listesi)*

Your SARIMA model is considered **successful** only if:
*(SARIMA modeliniz yalnızca şu durumlarda **başarılı** sayılır:)*

1.  [ ] **Lower Error:** MAE/RMSE is **significantly lower** (>10-15%) than the Naïve Seasonal Baseline.
    *(Daha Düşük Hata: MAE/RMSE, Saf Mevsimsel Referans'tan anlamlı derecede düşüktür.)*
2.  [ ] **Residuals are Random:** The errors look like **White Noise** (no leftover patterns).
    *(Hatalar Rastgele: Hatalar Beyaz Gürültü gibi görünür [arta kalan desen yok].)*
3.  [ ] **Captured Complexity:** The model predicts **trend changes** or **holiday spikes** that the baseline misses.
    *(Karmaşıklığı Yakalama: Model, referansın kaçırdığı trend değişimlerini veya tatil sıçramalarını tahmin eder.)*

> **🚀 Conclusion:** If the criteria above are met, the model has successfully learned **complex patterns** beyond simple repetition.
> *(Sonuç: Yukarıdaki kriterler karşılanıyorsa, model basit tekrarların ötesindeki karmaşık kalıpları başarıyla öğrenmiştir.)*


# 🏆 SARIMA Modelling: Step 6 - Evaluate, Diagnose & Grid Search
*(SARIMA Modelleme: Adım 6 - Değerlendirme, Tanılama ve Izgara Araması)*

Modelimizi eğittikten sonra, başarısını sayısal olarak kanıtlamalı ve en iyi parametre kombinasyonunu bulmalıyız.

---

## 📏 Evaluation Metrics
*(Değerlendirme Metrikleri)*

We use two primary criteria to rank our models:
*(Modellerimizi sıralamak için iki temel kriter kullanıyoruz:)*

1.  **AIC / BIC:** computed for each fitted model — **lower is better**.
    *(Her eğitilen model için hesaplanır — daha düşük olması daha iyidir.)*
2.  **Out-of-Sample MAE or RMSE:** evaluated on a hold-out set — **lower is better**.
    *(Dışarıda tutulan set [test seti] üzerinde değerlendirilen örneklem dışı hata — daha düşük olması daha iyidir.)*

> **🎯 The Sweet Spot:** When the same model ranks best on **both criteria** (lowest AIC and lowest MAE), you’ve found a well-tuned SARIMA ready for forecasting.

---

## 📊 Comparison: Plain ARIMA vs. SARIMA
*(Karşılaştırma: Düz ARIMA ve SARIMA)*

Before running the full grid search, let's look at the performance of our initial SARIMA configuration compared to the plain ARIMA we built earlier.



### Interpretation (Yorumlama)
* **Visual Performance:** Both the chart and the error numbers blow the plain ARIMA out of the water.
    *(Hem grafik hem de hata sayıları, düz ARIMA'yı belirgin şekilde geride bırakıyor.)*
* **Capturing Peaks:** The SARIMA traces the **weekly peaks** (*haftalık zirveleri takip ediyor*) instead of flattening them.
* **The "S" Factor:** Adding the **seasonal MA term** let the model capture the **7-day rhythm** that ARIMA kept missing.
    *(Mevsimsel MA terimini eklemek, modelin ARIMA'nın sürekli kaçırdığı 7 günlük ritmi yakalamasını sağladı.)*

---

## 💻 Try it Yourself: The Mini Grid-Search
*(Kendin Dene: Mini Izgara Araması)*

We’ve already fit one SARIMA configuration. Now, we turn that single-model notebook cell into a **mini grid-search** loop that tries every combination we sketched in Step 4.

### Python Implementation
*(Python Uygulaması)*

```python
from darts.models import ARIMA
from sklearn.metrics import mean_absolute_error

# Define the Grid
pdq      = [(4,0,0), (4,0,1), (5,0,0), (5,0,1), (6,0,0), (6,0,1)]
seasonal = [(0,1,0,7), (0,1,1,7)]

best_aic = float('inf')
best_cfg = None

# Loop through all combinations
for order in pdq:
    for s_order in seasonal:
        # Initialize and Fit
        model = ARIMA(order=order, seasonal_order=s_order)
        model.fit(train)

        # Predict and Evaluate
        pred = model.predict(len(test))
        mae  = mean_absolute_error(test.values.flatten(),
                                   pred.values.flatten())
        
        # Access AIC from the underlying statsmodels object
        aic  = model.model.aic          

        print(f"SARIMA{order}×{s_order}   AIC = {aic:.1f}   MAE = {mae:.2f}")

        # Track the Winner (Lowest AIC)
        if aic < best_aic:
            best_aic, best_cfg = aic, (order, s_order)

print(f"\n🏆 Best model: SARIMA{best_cfg[0]}×{best_cfg[1]}   AIC = {best_aic:.1f}")
```

# 🛠️ Common Troubleshooting & Key Takeaways
*(Yaygın Sorun Giderme ve Temel Çıkarımlar)*

SARIMA modelleri bazen beklenmedik sonuçlar verebilir. Aşağıdaki tablo, sık karşılaşılan belirtileri ve çözüm yollarını özetler.

---

### 🩺 Troubleshooting Matrix
*(Sorun Giderme Matrisi)*

| Symptom (Belirti) | Likely Issue (Olası Sorun) | Fix (Çözüm) |
| :--- | :--- | :--- |
| **ACF still spikes at seasonal lags**<br>*(ACF hala mevsimsel gecikmelerde sıçrama yapıyor)* | **D or (P,Q) too low**<br>*(D veya P/Q değerleri çok düşük)* | **Increase D to 1 or raise P/Q**<br>*(D'yi 1'e çıkarın veya P/Q değerlerini artırın.)* |
| **Model diverges / fails to converge**<br>*(Model ıraksıyor / yakınsamada başarısız oluyor)* | **Over-differenced or too many params**<br>*(Aşırı fark alma veya çok fazla parametre)* | **Reduce D or drop extra terms**<br>*(D'yi azaltın veya fazladan terimleri çıkarın.)* |
| **Forecast too flat**<br>*(Tahmin çok düz / değişim göstermiyor)* | **Count or intermittent data**<br>*(Sayım veya kesintili/aralıklı veri)* | **Try SARIMAX or Poisson-based model**<br>*(Dışsal değişkenli [exogenous regressors] SARIMAX veya Poisson tabanlı bir model deneyin.)* |

> ☝🏼 **Pro Tip:** Selecting a set of values for the model parameters is crucial. We highly recommend following a structured guide on **parameter selection**.
> *(Model parametreleri için bir değer seti seçmek hayati önem taşır. Parametre seçimi konusunda yapılandırılmış bir rehberi takip etmenizi şiddetle öneririz.)*

---

### 🔑 Key Take-aways
*(Temel Çıkarımlar)*

1.  **Equation:** **SARIMA = ARIMA + Seasonal Layer**. You only need it when ACF/PACF show repeating **seasonal spikes**.
    *(SARIMA = ARIMA + Mevsimsel Katman. Sadece ACF/PACF tekrarlayan mevsimsel sıçramalar gösterdiğinde buna ihtiyacınız vardır.)*
2.  **Season Length ($s$):** This is the **first big clue** and is almost always known from **business context** (weekly, monthly, quarterly).
    *(Sezon uzunluğu [s] ilk büyük ipucudur ve neredeyse her zaman iş bağlamından [haftalık, aylık, çeyreklik] bilinir.)*
3.  **Start Simple:** Start with $(p,d,q)$ from ARIMA + $(P,D,Q) = (1,1,1)$ and **iterate**.
    *(Basit başlayın: ARIMA'dan gelen [p,d,q] ve [P,D,Q] = [1,1,1] ile başlayıp tekrarlayarak ilerleyin.)*
4.  **Guidance:** Let **AIC/BIC** & **Validation Error** guide refinement, just like with ARIMA.
    *(Rehberlik: Tıpkı ARIMA'da olduğu gibi, iyileştirme sürecine AIC/BIC ve Doğrulama Hatasının rehberlik etmesine izin verin.)*

# ⚠️ Shortcomings of ARIMA & SARIMA Methods: A Critical Analysis
*(ARIMA ve SARIMA Yöntemlerinin Kısıtlamaları: Kritik Bir Analiz)*

ARIMA ve SARIMA, zaman serisi tahminciliğinin "altın standardı" olarak kabul edilse de, modern veri setlerinin karmaşıklığı karşısında belirgin yapısal zayıflıkları vardır. Aşağıdaki analiz, bu modellerin neden ve nerede başarısız olabileceğini teknik olarak detaylandırır.

---

### 1. 📉 Strict Stationarity Assumption
*(Katı Durgunluk Varsayımı)*

**The Limitation:** Both models fundamentally assume the data is **stationary** ($Mean$, $Variance$, and $Covariance$ do not change over time).
*(Kısıtlama: Her iki model de verinin temel olarak durgun olduğunu varsayar.)*

* **Technical Detail:**
    * **Over-differencing Risk:** Durgunluğu sağlamak için uygulanan fark alma (*differencing, $d$*) işlemi, serideki sinyali yok edebilir (*over-differencing*) ve modelin "hafızasını" yapay olarak kısaltabilir.
    * **Transformation Issues:** Varyansı sabitlemek için yapılan logaritmik veya Box-Cox dönüşümleri, tahminlerin geri dönüştürülmesinde (*inverse transformation*) sapmalara (*bias*) yol açabilir.
    * **Unit Root Problems:** Karmaşık serilerde, birim kök testleri (ADF, KPSS) çelişkili sonuçlar verebilir, bu da $d$ parametresinin yanlış seçilmesine neden olur.

### 2. 📏 Linearity Constraint
*(Doğrusallık Kısıtı)*

**The Limitation:** ARIMA is a **linear model**. It assumes the future is a linear combination of past values and past errors.
*(Kısıtlama: ARIMA doğrusal bir modeldir. Geleceğin, geçmiş değerlerin ve geçmiş hataların doğrusal bir kombinasyonu olduğunu varsayar.)*

* **Technical Detail:**
    * **Cannot Capture Volatility:** Finansal verilerde sık görülen **Heteroscedasticity** (*değişen varyans/oynaklık*) durumunu modelleyemez (Bunun için GARCH ailesi gerekir).
    * **Structural Breaks:** Verideki ani yapısal kırılmaları (örn. pandeminin başlaması) yönetemez; geçmişteki katsayıları geleceğe uygulamaya devam eder ve büyük hatalar üretir.
    * **Complex Interactions:** Değişkenler arasındaki doğrusal olmayan (*non-linear*) karmaşık ilişkileri (örn. doygunluk noktaları, eşik etkileri) yakalayamaz.

### 3. 🗓️ Rigidity in Seasonality (SARIMA)
*(Mevsimsellikte Katılık)*

**The Limitation:** SARIMA requires a **fixed integer seasonality** (*sabit tam sayı mevsimsellik*) and struggles with multiple seasonal patterns.
*(Kısıtlama: SARIMA sabit tam sayı mevsimsellik gerektirir ve çoklu mevsimsel desenlerde zorlanır.)*

* **Technical Detail:**
    * **Single Seasonality:** SARIMA sadece tek bir döngüyü (örn. sadece haftalık) modeller. Hem haftalık hem yıllık döngüsü olan verilerde (örn. günlük elektrik tüketimi) yetersiz kalır.
    * **Integer Constraint:** Periyot ($s$) tam sayı olmalıdır. Ancak gerçek hayatta bir yıl 52 hafta değil, **52.14** haftadır. Bu kayma, uzun vadeli tahminlerde faz hatasına (*phase shift*) neden olur.
    * **Dynamic Seasonality:** Mevsimselliğin zamanla değiştiği (*Modulated Seasonality*) durumlarda (örn. mevsimlerin kayması) model adapte olamaz.

### 4. 🐌 High Computational Cost
*(Yüksek Hesaplama Maliyeti)*

**The Limitation:** Fitting these models, especially with **Grid Search** (Auto-ARIMA), is computationally expensive ($O(N^2)$ or worse).
*(Kısıtlama: Bu modelleri eğitmek, özellikle Izgara Araması ile, hesaplama açısından pahalıdır.)*

* **Technical Detail:**
    * **Stepwise Estimation:** En iyi $(p,d,q)(P,D,Q)$ kombinasyonunu bulmak için modelin yüzlerce kez yeniden eğitilmesi ve her seferinde **AIC/BIC** hesaplanması gerekir.
    * **Large Lags:** Uzun mevsimsel periyotlar (örn. saatlik veride $s=168$ haftalık döngü) parametre uzayını patlatır ve optimizasyonun yakınsamamasına (*convergence failure*) neden olabilir.

### 5. 🚫 Limited Exogenous Support (ARIMA vs. ARIMAX)
*(Sınırlı Dışsal Destek)*

**The Limitation:** Standard ARIMA relies solely on endogenous (internal) data.
*(Kısıtlama: Standart ARIMA yalnızca içsel verilere dayanır.)*

* **Technical Detail:**
    * **Requirement for Future Values:** **ARIMAX** veya **SARIMAX** gibi uzantılar dışsal değişkenleri (*Exogenous Variables*) desteklese de, tahmin yapabilmek için bu dışsal değişkenlerin **gelecekteki değerlerini** de bilmeniz gerekir (örn. yarının satışını tahmin etmek için yarının hava durumunu bilmek gerekir). Bu, pratikte uygulanabilirliği zorlaştırır.

---

### 🔄 Alternative Approaches
*(Alternatif Yaklaşımlar)*

Bu kısıtlamaları aşmak için kullanılan diğer klasik ve modern yöntemler:

| Yöntem (Method) | Çözdüğü Sorun (Problem Solved) | Avantajı (Advantage) |
| :--- | :--- | :--- |
| **ETS (Exponential Smoothing)** | **Non-Stationarity & Seasonality** | Veriyi durağanlaştırmaya gerek duymaz; trend ve mevsimselliği doğrudan bileşen olarak modeller. |
| **Prophet (Meta)** | **Multiple Seasonality & Missing Data** | Birden fazla döngüyü (haftalık + yıllık) ve tatilleri esnek bir şekilde modeller; eksik verilere dayanıklıdır. |
| **TBATS** | **Complex/Non-Integer Seasonality** | Karmaşık ve tam sayı olmayan mevsimsellikleri (örn. 365.25) trigonometrik fonksiyonlarla çözer. |
| **Machine Learning (XGBoost/LightGBM)** | **Non-Linearity & Exogenous Vars** | Doğrusal olmayan ilişkileri mükemmel yakalar; dışsal değişkenleri yönetmek çok daha kolaydır. |

---

> **💡 Expert Verdict:**
> ARIMA/SARIMA are excellent for **short-term forecasting** on **simple, stable datasets** where interpretability is key. However, for complex, volatile, or multi-seasonal real-world data, exploring **Machine Learning** or hybrid methods (like Prophet) is often necessary.
> *(Uzman Kararı: ARIMA/SARIMA, yorumlanabilirliğin kilit olduğu basit ve kararlı veri setlerinde kısa vadeli tahminler için mükemmeldir. Ancak karmaşık, oynak veya çoklu mevsimselliğe sahip gerçek dünya verileri için Makine Öğrenmesi veya hibrit yöntemleri keşfetmek genellikle gereklidir.)*

# 📝 Quiz 3: ARIMA & SARIMA Conceptual Check
*(Quiz 3: ARIMA ve SARIMA Kavramsal Kontrol)*

Aşağıdaki sorular ve teknik açıklamalar, Zaman Serisi Modelleme konusundaki temel kavramları (ARIMA bileşenleri, SARIMA'nın farkı, Durağanlık ve Mevsimsellik) pekiştirmek için hazırlanmıştır.

---

### ❓ Question 1
**What does ARIMA stand for?**
*(ARIMA neyin kısaltmasıdır?)*

* **Correct Answer (Doğru Cevap):** **B - Autoregressive Integrated Moving Average**

> **💡 Technical Explanation (Teknik Açıklama):**
> ARIMA, zaman serisi analizinin üç temel bileşeninin matematiksel birleşimidir. İsim, modelin iç yapısını doğrudan tarif eder:
> * **AR (AutoRegressive):** Gelecekteki değerin, geçmiş değerlerin doğrusal bir kombinasyonu olduğunu varsayar ($p$).
> * **I (Integrated):** Seriyi **durağan** (*stationary*) hale getirmek için uygulanan fark alma işlemidir ($d$).
> * **MA (Moving Average):** Modelin tahmin hatasını simüle eder ($q$).
>
> *Özetle ARIMA, verinin kendi geçmişiyle (AR) ve geçmiş hata paylarıyla (MA) ilişkisini kurarken, trend etkisinden arındırılmış (I) bir yapı üzerinde çalışır.*

---

### ❓ Question 2
**Which of the following models is best suited for time-series data with strong seasonal patterns?**
*(Aşağıdaki modellerden hangisi güçlü mevsimsel kalıplara sahip zaman serisi verileri için en uygundur?)*

* **Correct Answer (Doğru Cevap):** **C - SARIMA**

> **💡 Technical Explanation (Teknik Açıklama):**
> Standart ARIMA modelleri "kısa vadeli hafızaya" sahiptir ve genel trendleri yakalar. Ancak veride düzenli aralıklarla tekrarlayan (örn. her Aralık ayında artan satışlar) güçlü bir **mevsimsellik** varsa yetersiz kalır.
>
> **SARIMA (Seasonal ARIMA)**, modele ikinci bir katman ekleyerek bu sorunu çözer. Sadece "dün" ($t-1$) ile değil, "geçen sezonun aynı dönemi" ($t-s$) ile de ilişki kurar.

---

### ❓ Question 3
**What is the purpose of the Seasonal Period (m) in SARIMA?**
*(SARIMA'da Mevsimsel Periyodun (m) amacı nedir?)*

* **Correct Answer (Doğru Cevap):** **A - To determine the length of the seasonal cycle**
*(Mevsimsel döngünün uzunluğunu belirlemek)*

> **💡 Technical Explanation (Teknik Açıklama):**
> Literatürde genellikle $s$ veya $m$ olarak gösterilen bu parametre, modelin "bir tam döngüyü tamamlamak için kaç zaman adımına ihtiyaç duyduğunu" tanımlar.
> * **Aylık Veri:** $m=12$ (Yıllık desen).
> * **Günlük Veri:** $m=7$ (Haftalık desen).
>
> Teknik olarak $m$, mevsimsel fark alma işleminde hangi gecikmedeki değerin çıkarılacağını ($y_t - y_{t-m}$) belirler.

---

### ❓ Question 4
**What is the primary difference between ARIMA and SARIMA?**
*(ARIMA ve SARIMA arasındaki temel fark nedir?)*

* **Correct Answer (Doğru Cevap):** **C - SARIMA includes additional terms to handle seasonal patterns, unlike ARIMA.**
*(SARIMA, ARIMA'dan farklı olarak mevsimsel kalıpları işlemek için ek terimler içerir.)*

> **💡 Technical Explanation (Teknik Açıklama):**
> Fark, matematiksel yapıdadır:
> * **ARIMA $(p,d,q)$:** Yalnızca mevsimsel olmayan otokorelasyonu modeller.
> * **SARIMA $(p,d,q) \times (P,D,Q)_m$:** ARIMA'yı kapsar ancak ona **çarpımsal** (*multiplicative*) bir yapı ekler.
>
> SARIMA, hem "dünkü hatayı" hem de "geçen yılın aynı günündeki hatayı" denkleme dahil ederek çok katmanlı serileri modeller.

---

### ❓ Question 5
**Which of the following statements about ARIMA is true?**
*(ARIMA ile ilgili aşağıdaki ifadelerden hangisi doğrudur?)*

* **Correct Answer (Doğru Cevap):** **B - ARIMA requires data to be stationary for accurate predictions.**
*(ARIMA, doğru tahminler için verilerin durağan olmasını gerektirir.)*

> **💡 Technical Explanation (Teknik Açıklama):**
> Bu, ARIMA'nın en temel varsayımıdır. **Durağanlık** (*Stationarity*); ortalama, varyans ve otokovaryansın zamanla değişmemesi anlamına gelir.
>
> ARIMA doğrusal bir model olduğu için, geçmişteki katsayıları geleceğe uygular. Eğer veride trend veya değişen varyans varsa, bu katsayılar geçersiz olur. Bu nedenle **"I" (Integrated)** bileşeni ile fark alınarak veri durağanlaştırılır.

---
---

# 🤖 Machine Learning for Time Series: Beyond Classical Methods
*(Zaman Serileri için Makine Öğrenimi: Klasik Yöntemlerin Ötesi)*

Geleneksel yöntemler (ARIMA, SARIMA, ETS), verileriniz "uslu durduğunda" (*low noise, clear seasonality*) oldukça güçlüdür. Ancak modern iş dünyası verileri genellikle karmaşıktır: binlerce ürün, onlarca dışsal sinyal (*exogenous signals*), rejim değişiklikleri (*regime shifts*) ve doğrusal olmayan etkileşimler içerir.

Bu doküman, Zaman Serisi tahminciliğinde neden ve nasıl **Makine Öğrenimi (ML)** yöntemlerine geçiş yapıldığını teknik derinlikle açıklar.

---

## 1. 🚀 Why Go Beyond "Classic" Forecasting?
*(Neden "Klasik" Tahminin Ötesine Geçmeliyiz?)*

ARIMA gibi modeller tek değişkenli (*univariate*) ve doğrusal (*linear*) varsayımlara dayanır. Makine öğrenimi modelleri ise şu avantajları sunar:

* **Non-Linear Patterns (Doğrusal Olmayan Desenler):** Satışlar ve fiyat arasındaki ilişki genellikle doğrusal değildir (örn. fiyat belli bir eşiği geçince satışlar çakılır). ML modelleri (özellikle ağaç tabanlılar ve sinir ağları) bu karmaşık etkileşimleri otomatik öğrenir.
* **Covariates & Exogenous Variables (Dışsal Değişkenler):** Klasik modellerde dışsal değişken eklemek (*ARIMAX*) zordur. ML modelleri hava durumu, promosyonlar, web trafiği gibi yüzlerce değişkeni (*feature*) zorlanmadan modele dahil eder.
* **Global Models & Cross-Learning (Global Modeller ve Çapraz Öğrenme):**
    * *Classic:* Her ürün için ayrı bir ARIMA modeli eğitilir (1000 ürün = 1000 model).
    * *ML:* Tek bir model, 1000 ürünün tamamından veriyi öğrenerek (*shared parameters*), geçmişi az olan yeni ürünler (*cold-start problem*) için bile diğer ürünlerden öğrendiği kalıpları kullanarak tahmin yapabilir.
* **Forecasting Strategy (Tahmin Stratejisi):** ML modelleri, hatayı adım adım biriktiren özyinelemeli (*recursive*) yöntemler yerine, geleceği doğrudan tahmin eden (*Direct Multi-step Forecast*) stratejileri daha iyi uygulayabilir.

---

## 2. 🔄 How ML Treats Time-Series Differently
*(ML Zaman Serilerine Nasıl Farklı Davranır?)*

ML algoritmaları (XGBoost, Neural Networks) zamanın sıralı yapısını doğrudan anlamazlar; veriyi onlara "öğretmemiz" gerekir.

| Konu (Topic) | Classical View (ARIMA/ETS) | ML View (Machine Learning) |
| :--- | :--- | :--- |
| **Input Shape**<br>*(Girdi Şekli)* | **1-D Sequence:** Veri sıralı bir vektördür. Model, $t$'deki değeri $t-1$'e bakarak tahmin eder. | **Tabular / Matrix:** Veri, denetimli öğrenme (*Supervised Learning*) problemine dönüştürülmelidir. Kayan pencereler (*Sliding Windows*) kullanılarak Özellik Matrisi ($X$) ve Hedef Vektörü ($y$) oluşturulur. |
| **Stationarity**<br>*(Durgunluk)* | **Critical:** Trend ve varyans sabitlenmelidir (Fark alma, Log dönüşümü). | **Flexible:** Ağaç tabanlı modeller durağan olmayan verilerle başa çıkabilir, ancak trendi "extrapolate" edemezler (eğitim setindeki max değerin üzerine çıkamazlar). Bu yüzden trendin arındırılması (*Detrending*) ML için de önemlidir. |
| **Model Family**<br>*(Model Ailesi)* | **Parametric:** Katsayıları bellidir, yorumlanabilirliği yüksektir. | **Non-Parametric / Black Box:** Esnektir (Ağaçlar, Sinir Ağları), çok karmaşık fonksiyonları öğrenir ancak "neden" sorusunu yanıtlamak (Feature Importance hariç) zordur. |
| **Forecast Strategy**<br>*(Tahmin Stratejisi)* | **Local (One-by-One):** Her seri için parametreler optimize edilir. | **Global:** Binlerce seri tek bir havuzda toplanır. Model genel kalıpları (*global structure*) öğrenir. |
| **Feature Engineering**<br>*(Özellik Müh.))* | **Minimal:** Lag ve Moving Average modelin içindedir. | **Heavy:** Lag, Rolling Mean, Calendar Features (Ay, Gün) manuel olarak üretilmelidir. |



---

## 3. 🤖 Common ML Models for Time-Series
*(Zaman Serileri için Yaygın ML Modelleri)*

### A. Tree-Based Ensembles (Ağaç Tabanlı Topluluklar)
* **Models:** XGBoost, LightGBM, CatBoost, Random Forest.
* **Mechanism:** Karar ağaçlarının topluluğunu oluşturarak tahmin yapar. Veriyi "bölerek" (*splitting*) öğrenir.
* **✅ Pros:**
    * Tablo şeklindeki verilerde (*Tabular Data*) ve heterojen özelliklerde (kategorik + sayısal) SOTA (*State-of-the-Art*) performansı verir.
    * Eksik verileri (*Missing Values*) doğal olarak yönetir.
    * Hızlıdır ve yorumlanabilir (*Feature Importance*).
* **❌ Cons:**
    * **Extrapolation Problem:** Eğitim verisinde gördüğü maksimum değerden daha yüksek bir değer tahmin edemez. Trend varsa mutlaka veri temizlenmeli veya lineer bir modelle birleştirilmelidir.
    * Manuel **Feature Engineering** (Lag, Rolling) gerektirir.

### B. Recurrent Neural Networks (RNNs)
* **Models:** LSTM (*Long Short-Term Memory*), GRU (*Gated Recurrent Unit*).
* **Mechanism:** Veriyi sıralı işler. "Hidden State" (Gizli Durum) sayesinde geçmiş bilgiyi hafızasında tutar.
* **✅ Pros:**
    * Uzun vadeli bağımlılıkları (*Long-term dependencies*) ve sıralı kalıpları doğal olarak öğrenir.
    * Manuel özellik mühendisliği ihtiyacı daha azdır.
* **❌ Cons:**
    * Sıralı işlem yaptığı için eğitimi yavaştır (*Non-parallelizable*).
    * "Vanishing Gradient" problemi yaşayabilir.



### C. Temporal Convolutional Networks (TCNs) & 1-D CNNs
* **Mechanism:** Görüntü işlemedeki CNN'lerin zaman serisine uyarlanmış halidir. Genişleyen evrişimler (*Dilated Convolutions*) kullanarak geniş bir geçmişe bakar.
* **✅ Pros:**
    * RNN'lerden çok daha hızlıdır (Paralel işlem yapılabilir).
    * Uzun dizilerde kararlıdır.
* **❌ Cons:**
    * Çok fazla veri gerektirir.



### D. Transformers / Attention Models
* **Models:** TFT (*Temporal Fusion Transformer*), Informer, Autoformer.
* **Mechanism:** "Attention" (*Dikkat*) mekanizması ile serinin hangi geçmiş noktalarının o anki tahmin için önemli olduğuna odaklanır.
* **✅ Pros:**
    * Uzun ufuklu tahminlerde (*Long-horizon forecasts*) şu anki en ileri teknolojidir.
    * Yorumlanabilirlik (*Interpretability*) sunar (TFT).
* **❌ Cons:**
    * GPU gücü ve çok büyük veri seti gerektirir.

### E. Hybrid Models (Statistical + ML)
* **Models:** Prophet (Trend + Seasonality + Regressors), ES-RNN, N-BEATS.
* **Mechanism:** İstatistiğin (Trend/Mevsimsellik ayrıştırma - STL) gücünü ML'in (Artıklar/Residuals üzerindeki öğrenme) gücüyle birleştirir.
* **✅ Pros:**
    * "Best of both worlds": Hem trendi iyi yönetir hem de karmaşık ilişkileri.

---

## 📅 Next Steps

1.  **Tree-Based Methods:** XGBoost kullanarak **Feature Engineering** (Özellik Mühendisliği) tekniklerine ve denetimli öğrenme dönüşümüne odaklanacağız.
2.  **Deep Learning:** LSTM ve RNN mimarilerini inceleyerek sıralı veri modellemeyi öğreneceğiz.


2.2. Creating Rolling Statistics: <img width="652" height="351" alt="image" src="https://github.com/user-attachments/assets/2fec113a-327a-4fcb-9eca-fc62a18f8205" />


# 🛠️ Feature Engineering for Time Series: Lags & Rolling Windows
*(Zaman Serileri için Özellik Mühendisliği: Gecikmeler ve Kayan Pencereler)*

Ağaç tabanlı algoritmalar (*Decision Trees, XGBoost, LightGBM, Random Forest*), her bir veri satırını zamandan bağımsız, tekil bir anlık görüntü (*independent snapshot*) olarak ele alır. Bu modeller, biz onlara dünün değerini ayrı bir sütun olarak vermediğimiz sürece "dünü" hatırlamazlar.

Modelin **momentum**, **ortalamaya dönüş** (*mean-reversion*) ve **mevsimsellik** (*seasonality*) gibi kalıpları öğrenebilmesi için, **Gecikme** (*Lag*) ve **Kayan Pencere** (*Rolling-Window*) özelliklerini kullanarak veriye zamansal bir hafıza enjekte etmeliyiz.

---

## 1. The Core Feature Types
*(Temel Özellik Türleri)*

Zaman serisi problemlerinde kullanılan en yaygın özelliklerin teknik özeti:

| Feature (Özellik) | What it captures (Neyi Yakalar?) | Typical Notation (Tipik Gösterim) |
| :--- | :--- | :--- |
| **Lag**<br>*(Gecikme)* | **Exact Memory:** $k$ adım önceki kesin değer. Modelin otokorelasyonu (*autocorrelation*) öğrenmesini sağlar.<br>*Örn: Dünkü satış.* | `lag_1`, `lag_7`, `lag_30` |
| **Rolling Mean**<br>*(Kayan Ortalama)* | **Local Trend / Level:** Belirli bir penceredeki ortalama seviye. Modelin "temel çizgisini" (*baseline*) belirler.<br>*Örn: Son 7 günün ortalaması.* | `roll_mean_7`, `ma_7` |
| **Rolling Std / Var**<br>*(Kayan Standart Sapma)* | **Volatility / Uncertainty:** Yakın geçmişteki oynaklık. Verinin ne kadar kararsız olduğunu gösterir.<br>*Örn: Son 14 gündeki değişim.* | `roll_std_14` |
| **Count-since-last-zero**<br>*(Son sıfırdan beri geçen süre)* | **Inter-arrival Info:** Kesintili talep (*intermittent demand*) veya nadir olaylar için geçen süreyi ölçer. | `days_since_last_sale` |



---

## 2. Step-by-Step Code Walkthrough
*(Adım Adım Kod Rehberi)*

### 2.1. Loading and Preparing Data
*(Veriyi Yükleme ve Hazırlama)*

Veri manipülasyonu için gerekli kütüphaneleri yükleyerek ve veri setini hazırlayarak başlıyoruz.

*(Not: Veri setinizin `datetime` indeksine sahip olduğundan ve frekansının (günlük/saatlik) düzgün ayarlandığından emin olun.)*

### 2.2. Feature Engineering for Machine Learning
*(Makine Öğrenimi için Özellik Mühendisliği)*

Gözetimli makine öğrenimi modelleri (*Supervised ML models*), girdi olarak bir özellik koleksiyonuna ihtiyaç duyar. XGBoost gibi modellerin zaman serisi tahmininde başarılı olabilmesi için ham veriyi anlamlı sinyallere dönüştürmeliyiz.

#### A. Creating Lag Features (Gecikme Özellikleri Oluşturma)
Lag özellikleri, zaman serisinin geçmiş değerlerini temsil eder. Bu özellikler, modelin "Dün ne oldu?", "Geçen hafta bugün ne oldu?" sorularına cevap vermesini sağlar.

```python
# Create lag features (e.g., sales from the previous day, previous week)
# Gecikme özellikleri oluşturma (örn. önceki günün, önceki haftanın satışları)

# t-1: Dünkü değer (En önemli özelliklerden biridir)
df_filtered['lag_1'] = df_filtered['unit_sales'].shift(1)

# t-7: Geçen hafta aynı günün değeri (Haftalık mevsimselliği yakalar)
df_filtered['lag_7'] = df_filtered['unit_sales'].shift(7)

# t-30: Geçen ayki değer (Aylık döngüyü yakalar)
df_filtered['lag_30'] = df_filtered['unit_sales'].shift(30)

# Drop NaN values created by shifting
# Lag işlemi (shift) ilk satırlarda NaN (boş) değerler oluşturur, bunları temizlemeliyiz.
df_filtered.dropna(inplace=True)
```

# 🚨 Critical Concept: Data Leakage & Rolling Features
*(Kritik Kavram: Veri Sızıntısı ve Kayan Özellikler)*

Zaman serisi özellik mühendisliğinde en sık yapılan ve en tehlikeli hata, gelecekteki bilgiyi modele sızdırmaktır.

---

## 1. The "Shift" Imperative: Avoiding Data Leakage
*(Kaydırma Zorunluluğu: Veri Sızıntısından Kaçınma)*

**The Rule:** You cannot use data from time $t$ to predict time $t$. You must use data from $t-1, t-2...$
*(Kural: t zamanını tahmin etmek için t zamanındaki veriyi kullanamazsınız. t-1, t-2... verilerini kullanmalısınız.)*

### 🚫 The Mistake (Hata)
If you calculate a rolling mean **without shifting**:
`df['rolling_mean'] = df['sales'].rolling(7).mean()`
* The mean for "Today" includes "Today's Sales".
* The model sees the answer (Target) inside the input feature.
* **Result:** 99% accuracy in training, massive failure in production. This is called **Look-ahead Bias**.

### ✅ The Fix (Çözüm)
Always **shift first**, then roll.
`df['rolling_mean'] = df['sales'].shift(1).rolling(7).mean()`
* Now, "Today's" rolling mean is actually calculated using data from "Yesterday" backwards.



---

## 2. Mastering Rolling Statistics
*(Kayan İstatistiklerde Ustalaşmak)*

Kayan istatistikler, veriye **bağlam** (*context*) kazandırır. Tek bir veri noktası gürültülü olabilir, ancak bir pencerenin özeti daha kararlı bir sinyaldir.



### 📈 Rolling Mean (Kayan Ortalama)
* **Purpose:** Smooths out short-term noise and captures the **Local Level/Trend**.
    *(Kısa vadeli gürültüyü yumuşatır ve Yerel Seviyeyi/Trendi yakalar.)*
* **Usage:** Acts as a "baseline" prediction. If the rolling mean is rising, the tree model sets a higher starting point for the forecast.

### 📊 Rolling Standard Deviation (Kayan Standart Sapma)
* **Purpose:** Measures **Volatility** (*Oynaklık*) and **Uncertainty** (*Belirsizlik*).
* **Usage:**
    * **Low Rolling Std:** The series is stable; the model can trust the `lag_1` value more.
    * **High Rolling Std:** The series is chaotic; the model may be more conservative or rely more on the rolling mean than the immediate lag.

---

## 💻 Correct Implementation Pattern
*(Doğru Uygulama Deseni)*

```python
# 1. Correct: Shift THEN Roll (No Leakage)
# Doğru: Önce Kaydır SONRA Yuvarla (Sızıntı Yok)
df['feature_roll_mean_7'] = df['sales'].shift(1).rolling(window=7).mean()

# 2. Wrong: Roll on current values (Leakage!)
# Yanlış: Mevcut değerler üzerinde yuvarla (Sızıntı!)
# df['feature_roll_mean_7'] = df['sales'].rolling(window=7).mean()  <-- DO NOT DO THIS

```

# 📅 Date-based Features & Model Logic
*(Tarih Bazlı Özellikler ve Model Mantığı)*

Gecikme (*lag*) ve kayan pencere (*rolling window*) özelliklerine ek olarak, zamanın kendisinden türetilen özellikler, modelin **periyodik desenleri** (*periodic patterns*) öğrenmesi için kritiktir.

Ağaç tabanlı modeller (*Tree-based models*), "zamanın akışını" bilmezler. Onlar için `2023-12-31` ile `2024-01-01` arasındaki ilişki belirsizdir. Bu ilişkiyi açık hale getirmek için zaman damgasını parçalarına ayırmalıyız.

---

### 1. 📆 Extracting Date Features
*(Tarih Özelliklerini Çıkarma)*

Zaman damgasından (*timestamp*) aşağıdaki özellikleri türeterek modelin mevsimselliği yakalamasını sağlarız:

```python
# Assuming the index is a datetime object
# İndeksin datetime objesi olduğunu varsayıyoruz

# 1. Basic Calendar Features (Temel Takvim Özellikleri)
df_filtered['day_of_week'] = df_filtered.index.dayofweek  # 0=Mon, 6=Sun
df_filtered['day_of_month'] = df_filtered.index.day
df_filtered['month'] = df_filtered.index.month
df_filtered['quarter'] = df_filtered.index.quarter
df_filtered['year'] = df_filtered.index.year

# 2. Boolean Flags (Mantıksal Bayraklar)
# Haftasonu etkisi için (Cumartesi/Pazar)
df_filtered['is_weekend'] = df_filtered.index.dayofweek.isin([5, 6]).astype(int)

# Yıl sonu/başı etkisi için
df_filtered['is_month_start'] = df_filtered.index.is_month_start.astype(int)
df_filtered['is_month_end'] = df_filtered.index.is_month_end.astype(int)
```

## 2. 🧠 How the Model Uses These Features
*(Model Bu Özellikleri Nasıl Kullanır?)*

Bir karar ağacı (*Decision Tree*), bu özellikleri kullanarak tahmin uzayını "böler" (*split*). Modelin bu özellikleri nasıl yorumladığını anlamak, model başarısını artırır.


| Feature Interaction (Özellik Etkileşimi) | Decision Tree Logic (Karar Ağacı Mantığı) |
| :--- | :--- |
| **Seasonal Spikes**<br>*(Mevsimsel Sıçramalar)* | **Logic:** `IF is_weekend == 1 THEN Predict High`<br><br>Ağaç, hafta sonu bayrağını gördüğünde dallanır ve Cumartesi/Pazar için temel tahmin değerini yükseltir (*boost*). |
| **Holiday Effects**<br>*(Tatil Etkileri)* | **Logic:** `IF month == 12 AND day > 20 THEN Predict Very High`<br><br>Model, Aralık ayının son günlerinde ortalamaların yüksek olduğunu öğrenerek yıl sonu yoğunluğunu yakalar. |
| **Local Smoothing**<br>*(Yerel Yumuşatma)* | **Logic:** `IF rolling_mean_7 > 50 THEN Predict > 45`<br><br>`rolling_mean` dinamik bir taban çizgisi (*dynamic baseline*) görevi görür. Trend yukarı yönlü ise, `lag_1` (dünkü satış) 0 olsa bile (stok bitmesi vb.), model ortalamaya güvenerek tahmini yüksek tutabilir. |
| **Volatility Awareness**<br>*(Volatilite Farkındalığı)* | **Logic:** `IF rolling_std_7 > 10 THEN Widen Interval`<br><br>Yüksek standart sapma, model için bir uyarıdır: "Büyük değişimler olabilir." Model, bu durumda daha muhafazakar davranabilir veya tahmin aralığını genişletebilir. |

### 🔑 Key Take-aways


Zaman serisi özellik mühendisliğinde kullanılan üç temel sütun türünün özeti:



* **Lag Columns** (*Gecikme Sütunları*):
    * **Provide Exact Memory** (*Kesin Hafıza Sağlar*).
    * **Captures Autocorrelation** (*Otokorelasyonu Yakalar*).
    * *Example:* "Yesterday's price directly affects today's price."
        *(Örnek: "Dünün fiyatı bugünün fiyatını doğrudan etkiler.")*

* **Rolling Columns** (*Kayan Sütunlar*):
    * **Provide Context** (*Bağlam Sağlar*).
    * **Captures Level** (*Seviye*) and **Volatility** (*Oynaklık*).
    * *Example:* "Is the general trend rising or falling this week?"
        *(Örnek: "Genel trend bu hafta yükseliyor mu yoksa düşüyor mu?")*

* **Calendar Columns** (*Takvim Sütunları*):
    * **Encode Periodic Effects** (*Periyodik Etkileri Kodlar*).
    * Eliminates the need for manual one-hot encoding for every single date.
        *(Her bir tarih için manuel one-hot encoding yapma ihtiyacını ortadan kaldırır.)*
    * *Example:* "Sales always spike on Fridays."
        *(Örnek: "Satışlar her zaman Cuma günleri zirve yapar.")*

---

### 🛡️ Safety First


> **⚠️ Avoid Leaking Future Info**
> *(Gelecek Bilgisini Sızdırmaktan Kaçının)*
>
> Always **shift** your data before calculating rolling statistics. If you include "today" in your "average of the last 7 days", the model will cheat by seeing the answer.
> *(Kayan istatistikleri hesaplamadan önce verilerinizi her zaman **kaydırın**. Eğer "son 7 günün ortalamasına" "bugünü" dahil ederseniz, model cevabı görerek hile yapar.)*
>
> **Correct Syntax:** `shift(1).rolling(7)`



# 🚀 XGBoost for Time-Series Forecasting
*(Zaman Serisi Tahmini İçin XGBoost)*

 <img width="905" height="443" alt="image" src="https://github.com/user-attachments/assets/572c3f18-add6-4445-913a-59eefed430c1" />

**XGBoost** (*Extreme Gradient Boosting*), son yıllarda veri bilimi dünyasını domine eden, özellikle yapılandırılmış/tablosal verilerde (*structured/tabular data*) gösterdiği üstün performansla bilinen güçlü bir makine öğrenimi algoritmasıdır.

Bu bölümde, XGBoost'un teknik altyapısını ve zaman serisi tahminciliğinde nasıl bir regresyon aracı olarak kullanıldığını inceleyeceğiz.

---

## 🧠 What is XGBoost?
*(XGBoost Nedir?)*

XGBoost, **Karar Ağaçları** (*Decision Trees*) temelli bir topluluk öğrenme (*Ensemble Learning*) yöntemidir. Temel mantığı **Gradient Boosting** prensibine dayanır:
* **Ensemble Strategy:** Zayıf öğrenicileri (*weak learners - sığ ağaçlar*) bir araya getirerek güçlü bir tahminci oluşturur.
* **Gradient Descent:** Her yeni ağaç, bir önceki ağacın yaptığı hataları (*residuals*) tahmin etmek ve düzeltmek üzerine eğitilir.

> **💡 Expert Note:** While Random Forest builds trees independent of each other (bagging), XGBoost builds trees sequentially (boosting), where each tree corrects the errors of the previous one.
> *(Uzman Notu: Rastgele Orman ağaçları birbirinden bağımsız kurarken [bagging], XGBoost ağaçları sıralı kurar [boosting]; her ağaç bir öncekinin hatasını düzeltir.)*

---

## 🌟 Why is XGBoost So Popular?
*(XGBoost Neden Bu Kadar Popüler?)*

XGBoost, sadece doğruluğu ile değil, mühendislik harikası optimizasyonları ile de öne çıkar.

| Feature (Özellik) | Technical Detail (Teknik Detay) |
| :--- | :--- |
| **Accuracy**<br>*(Doğruluk)* | Düşük varyans ve düşük yanlılık (*bias*) dengesini mükemmel kurar. Kaggle yarışmalarının vazgeçilmezidir. |
| **Speed & Performance**<br>*(Hız ve Performans)* | **Parallel Processing:** Ağaç oluşturma sırasında özellikleri paralel işler.<br>**Tree Pruning:** Ağacı geriye doğru budayarak (*max_depth*) gereksiz dalları temizler. |
| **Handling Missing Data**<br>*(Eksik Veri Yönetimi)* | **Sparsity-aware Split Finding:** Eksik değerler için "varsayılan" bir yön (*default direction*) öğrenir. Ön işleme yapmadan (*imputation*) eksik veriyi yönetebilir. |
| **Feature Importance**<br>*(Özellik Önemi)* | Veri setindeki hangi özelliklerin (örn. `lag_7`, `rolling_mean`) tahmine en çok katkı sağladığını otomatik olarak hesaplar (`gain`, `weight`, `cover`). |
| **Regularization**<br>*(Düzenlileştirme)* | **L1 (Lasso) & L2 (Ridge):** Aşırı öğrenmeyi (*overfitting*) engellemek için modelin karmaşıklığını cezalandıran yerleşik parametrelere sahiptir. |

---

## 🛠️ Building XGBoost Model for Demand Forecasting
*(Talep Tahmini İçin XGBoost Modeli Kurma)*

Zaman serisi tahminini bir **Denetimli Öğrenme** (*Supervised Learning*) problemi olarak ele alıyoruz.

### 1. Splitting Data: The Temporal Split
*(Veriyi Bölme: Zamansal Ayrım)*

Zaman serilerinde rastgele bölme (*random shuffle*) **yapılamaz**. Geleceği geçmişle tahmin etmeliyiz.

* **Training Set:** Geçmiş veriler (örn. 2020-2022).
* **Testing Set:** En güncel veriler (örn. 2023).
* **Goal:** Prevent **Data Leakage** (*Hedef: Veri sızıntısını önlemek*).

### 2. Implementing XGBoost
*(XGBoost Uygulaması)*

Modeli kurarken, özellik mühendisliği aşamasında ürettiğimiz `lag` ve `rolling` özelliklerini girdi olarak kullanırız.

```python
import xgboost as xgb

# Define the model with key hyperparameters
model = xgb.XGBRegressor(
    n_estimators=1000,     # Number of trees (Ağaç sayısı)
    learning_rate=0.01,    # Step size shrinkage (Öğrenme oranı)
    max_depth=5,           # Depth of trees (Ağaç derinliği)
    subsample=0.8,         # Row sampling (Satır örnekleme)
    colsample_bytree=0.8,  # Feature sampling (Özellik örnekleme)
    objective='reg:squarederror' # Loss function (Kayıp fonksiyonu)
)

# Train the model
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          early_stopping_rounds=50, # Stop if validation score doesn't improve
          verbose=False)
 
```

## 3. Performansı Görselleştirme: Gerçek ve Tahmin (Visualizing Performance: Actual vs Predicted)

Modeli test seti üzerinde değerlendirdiğimizde genellikle şu davranışı görürüz:

* **Başarı (Success):** Model haftalık zirve ve dip zamanlamalarını oldukça iyi takip eder; 7 günlük ritmi yakalar (The model tracks the timing of weekly peaks and troughs fairly well).
* **Kısıt (Limitation):** Genlikler genellikle sapar. Model uç değerleri/sıçramaları yumuşatma eğilimindedir (The amplitudes are often off. It tends to smooth out extreme spikes).
* **Sebep (Reason):** Ağaç tabanlı modeller (Tree-based models), eğitimde gördüklerinin ötesindeki değerleri tahmin edemez (cannot extrapolate). Bir yaprak düğümün ortalamasını tahmin ederler (They predict the average of a leaf node).

## 4. Değerlendirme Metrikleri: MSE ve MAE (Evaluation Metrics: MSE vs MAE)

### Optimizasyon Metriği - Eğitim (Optimization Metric - Training)
`objective='reg:squarederror'`

* MSE/RMSE'yi minimize eder.
* Büyük hataları, kareli terim nedeniyle ağır cezalandırır (Penalizes large errors heavily).
* Gradyan İnişi (Gradient Descent) için türevi alınabilir (Differentiable).

### Raporlama Metriği - Değerlendirme (Reporting Metric - Evaluation)
**MAE (Ortalama Mutlak Hata / Mean Absolute Error)**

* Performansı paydaşlara (stakeholders) raporlamak için kullanılır.
* Hatayı yorumlanabilir birimlerle ifade eder; örneğin, *"Ortalama ±50 birim sapıyoruz"* (Expresses error in interpretable units).

---

## 🔑 Temel Çıkarımlar (Key Takeaways)

* **Özellik Mühendisliği Kraldır (Feature Engineering is King):** XGBoost "zamanı" göremez. Zamansal bağımlılıkları (temporal dependencies) anlamak için tamamen oluşturduğumuz gecikme (lag) ve yuvarlanan (rolling) özelliklere güvenir.
* **Esneklik (Flexibility):** Doğrusal olmayan ilişkileri (non-linear relationships) ve etkileşimleri doğrusal (linear) ARIMA modellerinden daha iyi yönetir.
* **Ekstrapolasyon Uyarısı (Extrapolation Warning):** XGBoost, eğitim verisi aralığının çok üzerine çıkan veya altına inen bir trendi tahmin edemez; doğrusal regresyonun aksine (cannot predict a trend that goes significantly higher/lower than the training data range).


# Quiz 4 Solution: XGBoost Fundamentals

Aşağıda XGBoost ile ilgili temel kavramları içeren Quiz 4'ün çözümleri ve teknik açıklamaları yer almaktadır.

---

### 1. What does XGBoost stand for?
**(XGBoost neyin kısaltmasıdır?)**

* [ ] A - Extended Gradient Boosting
* [x] **B - Extreme Gradient Boosting**
* [ ] C - Experimental Gradient Boosting
* [ ] D - Exponential Gradient Boosting

> **Technical Note:** "Extreme" refers to the computational efficiency and engineering goal of pushing the limits of computing resources for boosted tree algorithms.
> *(Teknik Not: "Extreme", artırılmış ağaç algoritmaları için hesaplama kaynaklarının sınırlarını zorlayan mühendislik hedefine ve verimliliğe atıfta bulunur.)*

### 2. What is the primary function of boosting in XGBoost?
**(XGBoost'ta boosting'in birincil işlevi nedir?)**

* [ ] A - Creating a single deep tree to model the data
* [ ] B - Removing irrelevant features from the dataset
* [x] **C - Correcting errors made by previous trees in the model**
* [ ] D - Optimizing the training process by skipping some data points

> **Technical Note:** Boosting is a sequential ensemble technique where new trees are added to predict and correct the residuals (errors) of prior trees.
> *(Teknik Not: Boosting, yeni ağaçların önceki ağaçların kalıntılarını [hatalarını] tahmin etmek ve düzeltmek için eklendiği sıralı bir topluluk tekniğidir.)*

### 3. Why is XGBoost popular for machine learning tasks?
**(XGBoost makine öğrenimi görevleri için neden popülerdir?)**

* [ ] A - It automatically performs feature scaling and normalization
* [ ] B - It only works with structured/tabular data
* [ ] C - It is a deep learning algorithm optimized for unstructured data
* [x] **D - It provides accurate predictions, is fast, and handles missing data well**

> **Technical Note:** Its popularity stems from system optimization (speed), regularization (accuracy), and sparsity-aware algorithms (handling missing data natively).
> *(Teknik Not: Popülaritesi; sistem optimizasyonu [hız], düzenlileştirme [doğruluk] ve seyrekliğe duyarlı algoritmalarından [kayıp veriyi doğal olarak işleme] kaynaklanır.)*

### 4. What type of data is XGBoost best suited for?
**(XGBoost hangi veri türü için en uygundur?)**

* [ ] A - Image data
* [x] **B - Structured/tabular data**
* [ ] C - Text data
* [ ] D - Time-series data only

> **Technical Note:** Tree-based models excel at splitting heterogeneous features found in tabular data, whereas Deep Learning is better for unstructured data like images.
> *(Teknik Not: Ağaç tabanlı modeller, tablosal verilerdeki heterojen özellikleri bölmede mükemmeldir; Derin Öğrenme ise görüntüler gibi yapılandırılmamış verilerde daha iyidir.)*

### 5. How does XGBoost make its final predictions?
**(XGBoost son tahminlerini nasıl yapar?)**

* [ ] A - By selecting the best-performing single tree
* [ ] B - By averaging the predictions from all trees
* [x] **C - By combining the results of all trees, with each tree correcting the previous one**
* [ ] D - By using only the last tree in the boosting process

> **Technical Note:** It uses an additive strategy where the final prediction is the sum of outputs from all trees ($\sum f_k(x)$).
> *(Teknik Not: Son tahminin tüm ağaçların çıktılarının toplamı olduğu toplamsal [additive] bir strateji kullanır.)*

### 6. Which of the following is a key advantage of XGBoost?
**(Aşağıdakilerden hangisi XGBoost'un temel bir avantajıdır?)**

* [ ] A - It requires no feature engineering to produce accurate results
* [x] **B - It can handle missing data naturally without preprocessing**
* [ ] C - It is only used for classification tasks
* [ ] D - It does not require regularization to control overfitting

> **Technical Note:** XGBoost employs a "Sparsity-aware Split Finding" algorithm that learns the optimal default direction for missing values during training.
> *(Teknik Not: XGBoost, eğitim sırasında kayıp değerler için en uygun varsayılan yönü öğrenen "Seyrekliğe Duyarlı Bölme Bulma" algoritmasını kullanır.)*

### 7. Which of the following describes feature importance in XGBoost?
**(Aşağıdakilerden hangisi XGBoost'ta özellik önemini tanımlar?)**

* [ ] A - It automatically removes unimportant features from the dataset
* [x] **B - It identifies the most significant features in driving predictions**
* [ ] C - It assigns equal importance to all features in the dataset
* [ ] D - It requires manual calculation by the data scientist

> **Technical Note:** Importance is calculated based on metrics like "Gain" (how much a feature improves the tree's accuracy) or "Cover" (number of samples affected).
> *(Teknik Not: Önem; "Kazanç" [bir özelliğin ağacın doğruluğunu ne kadar artırdığı] veya "Kapsama" [etkilenen örnek sayısı] gibi metriklere göre hesaplanır.)*



# 🧠 Introduction to Deep Learning: A Time-Series Perspective


Derin Öğrenme (Deep Learning - DL), veriden öğrenmek ve modellemek için yapay sinir ağlarını (Artificial Neural Networks - ANNs) kullanan Makine Öğreniminin (Machine Learning - ML) gelişmiş bir alt alanıdır. Bu ağlar, karmaşık desenleri tanımlamada ve bu verilere dayanarak karar vermede mükemmeldir.

Bir Zaman Serisi (Time-Series) uzmanı gözüyle baktığımızda DL, geleneksel istatistiksel yöntemlerin (ARIMA, Exponential Smoothing) tıkandığı noktalarda devreye girer. Özellikle ham veriden önemli özellikleri (features) otomatik olarak çıkarma yeteneği; zaman serileri, görüntü tanıma, dil işleme gibi alanlarda devrim yaratmıştır.

---

## 🏗️ (Deep) Neural Networks: The Architecture
**((Derin) Sinir Ağları: Mimari)**

Tipik bir sinir ağı (Neural Network - NN), veriyi işlemek ve öğrenmek için birlikte çalışan, nöron (neuron) adı verilen birbirine bağlı katmanlardan oluşur. Bu yapı, insan beyninin çalışma prensibinden esinlenmiştir ancak matematiksel bir optimizasyon makinesidir.


 <img width="914" height="430" alt="image" src="https://github.com/user-attachments/assets/1826729f-6ed9-49b8-a6e5-c02c3d9bcae7" />


### 1. Input Layer (Girdi Katmanı)
Modelin dünyaya açılan kapısıdır. Ham özellikler modele buradan girer.
* **Genel:** Pikseller, kelimeler, sensör okumaları.
* **Time-Series Özel:** Gecikmeli değerler (lags), kayan pencere istatistikleri (rolling stats), takvim özellikleri (calendar features) veya ham sıralı veriler (t, t-1, t-2...).

### 2. Hidden Layers (Gizli Katmanlar ≥ 1)
Burası "sihrin" gerçekleştiği yerdir. Girdilerin ağırlıklandırılıp (weighted) işlendiği katmanlardır.
* Eğer birden fazla gizli katman varsa, ağa **Derin Sinir Ağı (Deep Neural Network - DNN)** denir.
* **Aktivasyon Fonksiyonları (Activation Functions):** Her nöronun çıktısı, `ReLU`, `Sigmoid` veya `Tanh` gibi doğrusal olmayan (non-linear) fonksiyonlardan geçirilir. Bu, ağın karmaşık, eğrisel ve çok boyutlu ilişkileri öğrenmesini sağlar.
* *Zaman serilerinde bu katmanlar genellikle LSTM veya GRU hücreleri ya da 1D-CNN filtreleri içerir.*

### 3. Output Layer (Çıktı Katmanı)
Son aktivasyonlar tahminlere dönüştürülür.
* **Regresyon (Time-Series):** Genellikle tek bir nöron (gelecekteki satış miktarı, sıcaklık vb.) ve `Linear` aktivasyon.
* **Sınıflandırma:** Olasılıklar (Softmax) veya sınıflar.

> 📌 **Technical Insight:** Sinir ağının katmanları, girdi verisini bir dizi doğrusal olmayan dönüşüm (nonlinear transformations) yoluyla işler. Bu, ağın "Evrensel Yaklaşıklık Teoremi" (Universal Approximation Theorem) sayesinde teorik olarak herhangi bir fonksiyonu öğrenebilmesine olanak tanır.

---

## ⚙️ How Deep Neural Networks Learn
**(Derin Sinir Ağları Nasıl Öğrenir?)**

Bir derin öğrenme modelinin faydalı olabilmesi için önce eğitilmesi (training) gerekir. Bu süreç, milyonlarca parametrenin (ağırlıklar ve sapmalar) optimize edildiği iteratif bir döngüdür.

### 1. Forward Propagation (İleri Yayılım)
Veri, girdi katmanından çıktı katmanına doğru akar. Her katman veriyi işler (matris çarpımı + aktivasyon) ve bir sonrakine iletir. Başlangıçta ağırlıklar rastgeledir, bu yüzden ilk tahminler tamamen yanlıştır.

### 2. Error Calculation: Loss Function (Hata Hesaplama: Kayıp Fonksiyonu)
Ağ bir tahminde bulunduğunda, model bu tahminin gerçek değerden (Ground Truth) ne kadar uzak olduğunu hesaplar.
* **Time-Series için:** Genellikle `MSE` (Mean Squared Error), `MAE` (Mean Absolute Error) veya olasılıksal tahminler için `Quantile Loss` kullanılır.
* **Classification için:** `Cross-Entropy` yaygındır.

### 3. Back-propagation (Geri Yayılım)
Bu adım, öğrenmenin kalbidir. Kayıp (Loss) hesaplandıktan sonra, ağ hataları azaltmak için ağırlıkları (weights) nasıl ayarlaması gerektiğini matematiksel olarak hesaplar.
* Hata, çıktıdan girdiye doğru geriye yayılır.
* Her bir ağırlığın hataya ne kadar katkıda bulunduğu (kısmi türevler/gradyanlar) zincir kuralı (chain rule) ile hesaplanır.



### 4. Gradient Descent & Optimization (Gradyan İnişi ve Optimizasyon)
Model, ağırlıkları güncellemek için optimizasyon algoritmaları kullanır.
* Amaç, kayıp fonksiyonunun en dik iniş yönünü (negatif gradyan) takip ederek global minimuma ulaşmaktır.
* **Algorithm:** Klasik `SGD` (Stochastic Gradient Descent) yerine, günümüzde genellikle adaptif öğrenme oranına sahip `Adam` (Adaptive Moment Estimation) optimizer tercih edilir.

> 📌 **Training Note:** Eğitim genellikle büyük veri setleri ve birçok iterasyon (veya **epochs**) gerektirir. Aşırı öğrenmeyi (Overfitting) önlemek için `Dropout` veya `Early Stopping` gibi teknikler de bu sürece dahil edilir.

---

## 🛠️ Common Deep Learning Frameworks
**(Yaygın Derin Öğrenme Çatıları)**

Derin öğrenmeye başlamak, güçlü kütüphaneler sayesinde artık çok daha kolaydır. Bu çerçeveler, arkadaki karmaşık türev ve matris işlemlerini (autograd) otomatik halleder.

| Framework | Açıklama (Description) | Kullanım Alanı (Use Case) |
| :--- | :--- | :--- |
| **TensorFlow (Keras)** | Google tarafından geliştirildi. Hem araştırma hem de üretim (production) ortamlarında güçlüdür. `Keras` API'si ile çok hızlı prototip üretilir. | Endüstriyel dağıtım, Mobil (TF Lite). |
| **PyTorch** | Meta (Facebook) tarafından geliştirildi. Esnekliği ve kullanım kolaylığı ile bilinir. Dinamik hesaplama grafikleri (dynamic computation graphs) hata ayıklamayı kolaylaştırır. | Akademik araştırma, Modern Time-Series kütüphaneleri (PyTorch Forecasting, Darts). |

> **Expert Opinion:** Geçmişte TensorFlow daha yaygındı, ancak modern araştırmalarda ve özellikle zaman serisi için geliştirilen yeni mimarilerde (Transformer tabanlı modeller) **PyTorch** fiili standart haline gelmiştir. "Under the hood" (kaputun altında) çalışan birçok kütüphane PyTorch kullanır.

---

## 🚀 Key Benefits of Deep Learning in Time-Series


Geleneksel makine öğrenimine ve klasik istatistiğe (ARIMA vb.) kıyasla DL'in öne çıktığı noktalar:

1.  **Automated Feature Extraction (Otomatik Özellik Çıkarımı):**
    * Geleneksel yöntemlerde trendi, mevsimselliği ve döngüleri elle ayrıştırmanız gerekir. DL (özellikle CNN ve RNN'ler), ham veriden bu kalıpları otomatik olarak öğrenir.
2.  **Handling Complex & High-Dimensional Data (Karmaşık ve Çok Boyutlu Veri Yönetimi):**
    * DL, yapılandırılmamış verilerle (görüntü, metin) çalışabildiği gibi, zaman serilerinde **Global Modeller** (Global Models) oluşturabilir. Yani, 1000 farklı ürünün satış verisini tek bir modelde eğiterek, ürünler arası ilişkileri (cross-learning) öğrenebilir.
3.  **Non-Linearity & Generalization (Doğrusallık Dışı ve Genelleştirme):**
    * Zaman serileri nadiren doğrusaldır. DL, karmaşık, kaotik ve doğrusal olmayan ilişkileri modellemede ve görülmemiş verilere genellemede (generalization) üstündür.

---

## ⚠️ Challenges in Deep Learning


Pratikte DL kullanmak bazı engelleri aşmayı gerektirir:

* **Data Requirements (Veri Gereksinimleri):** DL modelleri "veri açlığı" çeker. Yüksek performans için genellikle büyük miktarda etiketli geçmiş veriye ihtiyaç duyarlar. Az veriyle (Small Data) klasik yöntemler bazen daha iyi çalışabilir.
* **Computational Resources (Hesaplama Kaynakları):** Derin ağları eğitmek işlemci gücü ister. GPU'lar (Graphics Processing Units) veya TPU'lar olmadan büyük modelleri eğitmek günler sürebilir.
* **Interpretability (Yorumlanabilirlik):** Derin ağlar, özellikle çok katmanlı yapılar, genellikle "Kara Kutu" (Black Box) olarak adlandırılır. Bir tahminin *neden* yapıldığını anlamak (Feature Importance), karar ağaçlarına göre daha zordur. Finans veya sağlık gibi alanlarda bu bir risk faktörüdür (gerçi `TFT - Temporal Fusion Transformer` gibi modern mimariler bunu çözmeye odaklanmaktadır).

---

## 🏁 Conclusion


Derin Öğrenme, makine öğrenimi görevlerine yaklaşımımızı kökten değiştirdi. Büyük veri setleriyle başa çıkma ve ham veriden anlamlı desenler çıkarma yeteneği, onu modern veri biliminin en güçlü aracı yapar.

Sinir ağlarını kullanarak, DL modelleri karmaşık temsilleri otomatik olarak öğrenir. Bu derste/kapsamda, genel DL mimarilerinin ötesine geçip, Zaman Serisi kullanım durumlarımız (Time Series Use Cases) için özelleşmiş mimarilere odaklanacağız:
* **RNNs (Recurrent Neural Networks - LSTM/GRU):** Sıralı bağımlılıkları hatırlamak için.
* **1D-CNNs:** Zaman içindeki yerel desenleri yakalamak için.
* **Transformers:** Dikkat mekanizması (Attention) ile uzun vadeli ilişkileri modellemek için.
 



# 🔄 Recurrent Neural Networks (RNNs) for Time-Series
**(Zaman Serileri için Tekrarlayan Sinir Ağları)**

Zaman serisi tahmini (Time-Series Forecasting) için kullanılan Derin Öğrenme mimarileri arasında en temel ve yaygın bilinen iki yapı **Recurrent Neural Networks (RNNs)** ve onların gelişmiş versiyonu olan **Long Short-Term Memory Networks (LSTMs)**'dir.

Bu bölümde, modern sıralı modellemenin (sequential modeling) atası olan RNN'lerin teknik altyapısını ve sınırlamalarını inceleyeceğiz.

---

## 🧠 Recurrent Neural Networks (RNNs)

**Recurrent Neural Networks (RNNs)**, önceki girdilerin bir "hafızasını" (memory) koruyarak sıralı verileri (sequential data) işlemek üzere tasarlanmış özel bir sinir ağı sınıfıdır.

<img width="661" height="310" alt="image" src="https://github.com/user-attachments/assets/3033e0fc-264f-46b2-ba4f-9b7ebfe3f977" />

Bu hafıza yeteneği, bir zaman adımındaki (time step) tahminin, önceki zaman adımlarındaki verilere bağlı olduğu zaman serisi tahmini gibi görevler için onları ideal kılar. Geleneksel İleri Beslemeli (Feed-Forward) ağların aksine, RNN'ler zamanı bir boyut olarak kabul eder.

### Geleneksel Ağlardan Farkı (The Difference)

* **Traditional Neural Networks (Feed-Forward):** Girdiler birbirinden bağımsız kabul edilir (inputs are treated independently). Örneğin, bir kedi fotoğrafını tanıyan model, bir önceki fotoğrafta ne gördüğünü hatırlamaz. Veri akışı tek yönlüdür: Girdi -> Gizli Katman -> Çıktı.
* **Recurrent Neural Networks (RNNs):** Önceki zaman adımlarından gelen çıktıyı (output), mevcut zaman adımı için girdinin bir parçası olarak kullanır. Bu, RNN'lerin zaman içindeki kalıpları (patterns over time) yakalamasını sağlar.

> 📌 **Expert Note:** RNN'leri, kendi çıktısını bir sonraki adımda kendine girdi olarak veren "döngüsel" (looping) bir yapı olarak düşünebilirsiniz. Bu yapı "açıldığında" (unfolded), her zaman adımı için birbirinin kopyası olan bir ağ zinciri ortaya çıkar.



---

## ⚙️ How RNNs Work: The "Hidden State"
**(RNN'ler Nasıl Çalışır: "Gizli Durum")**

Bir RNN'deki her düğüm (node/neuron) sadece mevcut girdiyi işlemekle kalmaz, aynı zamanda ağın önceki durumunu da hatırlar. Bu hafıza mekanizmasına **Gizli Durum (Hidden State)** denir.

Matematiksel olarak süreç şu şekilde işler:

1.  **Input ($x_t$):** $t$ zamanındaki veri.
2.  **Previous Hidden State ($h_{t-1}$):** Ağın $t-1$ anındaki hafızası.
3.  **Current Hidden State ($h_t$):** Ağ, mevcut girdiyi ve eski hafızayı birleştirerek yeni bir hafıza durumu oluşturur.
    * Formül: $h_t = \tanh(W_h \cdot h_{t-1} + W_x \cdot x_t)$
4.  **Output ($y_t$):** Yeni gizli durum kullanılarak o anki tahmin yapılır.

Bu mekanizma, RNN'lerin tarihsel verilere (historical data) dayalı tahminler yapmasını sağlar ve onları zaman serisi görevleri için doğal bir seçim haline getirir.

### Comparison: FFN vs RNN
**(Karşılaştırma: İleri Beslemeli vs Tekrarlayan Ağlar)**

* **(a) Fully-Connected (Dense) Networks:** Her girdi bağımsızdır. Zaman kavramı yoktur. $x \to y$
* **(b) Recurrent Networks:** Girdiler sıralıdır. Şimdiki karar, geçmişe bağlıdır. $x_{t}, h_{t-1} \to y_{t}$

---

## ⚠️ The Limitation: Vanishing Gradient Problem
**(Kısıt: Kaybolan Gradyan Problemi)**

Teoride RNN'ler, sonsuz geçmişe bakabilir. Ancak pratikte, temel RNN'lerin (Vanilla RNNs) çok ciddi bir sınırlaması vardır: **Vanishing Gradient Problem (Kaybolan Gradyan Problemi)**.

### Bu Problem Nedir?
Ağı eğitirken **Zaman İçinde Geri Yayılım (Backpropagation Through Time - BPTT)** algoritmasını kullanırız. Hata (loss), zamandan geriye doğru (bugünden geçmişe) yayılırken ağırlıklar (weights) güncellenir.

* Eğer ağırlıklar küçükse (< 1), hata geriye doğru her adımda çarpılarak küçülür.
* Zincirleme çarpım sonucu (örn. $0.9 \times 0.9 \times 0.9 \dots$), gradyanlar hızla sıfıra yaklaşır.
* **Sonuç:** Ağ, serinin başındaki (uzak geçmişteki) verileri öğrenemez. Ağırlıklar güncellenemediği için ağın "hafızası" kısalır.

> **Impact:** Ağ sadece yakın geçmişe (short-term memory) odaklanır, uzun vadeli bağımlılıkları (long-term dependencies) öğrenemez. Örneğin, geçen yılki bir trendin bugünkü satışı etkilediğini RNN ile modellemek çok zordur.

---

## ⏭️ Why We Move to LSTMs
**(Neden LSTM'lere Geçiyoruz?)**

Temel RNN'lerin uzun vadeli bilgiyi "unutma" eğilimi, karmaşık zaman serileri için yetersiz kalmalarına neden olur. İşte bu yüzden, RNN'ler üzerine uzun bir sohbeti atlayıp, doğrudan bu problemin çözümü olan **LSTM (Long Short-Term Memory)** ve **GRU (Gated Recurrent Unit)** ağlarına geçiyoruz!

LSTM'ler, içerdikleri özel "kapı" (gate) mekanizmaları sayesinde hangi bilginin saklanacağını ve hangisinin unutulacağını seçerek kaybolan gradyan problemini çözerler.



# 🧠 Long Short-Term Memory Networks (LSTMs) for Time-Series
**(Zaman Serileri için Uzun Kısa-Vadeli Hafıza Ağları)**

 <img width="727" height="391" alt="image" src="https://github.com/user-attachments/assets/87415182-6893-444f-abce-092e982be577" />

Zaman serisi tahminciliğinde (Time-Series Forecasting) "altın standart" olarak kabul edilen mimarilerden biri **LSTM (Long Short-Term Memory)** ağlarıdır.

LSTM'ler, standart RNN'lerin (Recurrent Neural Networks) en büyük zaafı olan **Kaybolan Gradyan Problemini (Vanishing Gradient Problem)** çözmek için tasarlanmış özelleşmiş bir mimaridir. Standart RNN'ler zaman adımları arttıkça geçmişi unuturken, LSTM'ler **Kapılar (Gates)** adı verilen mekanizmalar sayesinde bilginin akışını kontrol eder. Bu sayede ağ, uzun diziler boyunca hangi bilginin saklanacağını, hangisinin unutulacağını ve hangisinin bir sonraki adıma aktarılacağını "öğrenir".



---

## 🚀 Key Benefits for Time-Series
**(Zaman Serileri İçin Temel Faydalar)**

Neden klasik yöntemler veya basit RNN'ler yerine LSTM kullanmalıyız?

1.  **Handling Long-Term Dependencies (Uzun Vadeli Bağımlılıkları Yönetme):**
    * Tahminlerin çok eski verilere dayandığı durumlar için idealdir.
    * *Örnek:* Bir perakendeci için bugünkü satışlar, 12 ay önceki "yıllık mevsimsellikten" (seasonal effects) etkilenebilir. LSTM bu bilgiyi taşıyabilir.
2.  **Complex & Non-Linear Patterns (Karmaşık ve Doğrusal Olmayan Desenler):**
    * Verilerdeki karmaşık, çok adımlı ve doğrusal olmayan kalıpları yakalar.
    * Geleneksel modellerin (ARIMA) veya basit RNN'lerin modellemekte zorlandığı ani değişimleri (shocks) ve rejim değişikliklerini yönetebilir.
3.  **Variable Length Sequences (Değişken Uzunluklu Diziler):**
    * Sabit uzunluklu girdilere sıkışıp kalmaz, farklı uzunluktaki tarihsel verileri işleyebilir.

> 📊 **Retail Forecasting Context:** LSTM ağları perakende tahminciliği için özellikle etkilidir çünkü geçmiş verilerden bilgiyi saklayarak hem kısa vadeli dalgalanmaları (haftalık döngüler) hem de uzun vadeli trendleri (yıllık büyüme) aynı anda yakalayabilirler. Promosyonlar, tatiller ve mevsimsellik gibi faktörlerin satış üzerindeki etkisini öğrenirler.

---

## ⚙️ Key Components of LSTMs: The "Mini-Factory"
**(LSTM'in Temel Bileşenleri: "Mini Fabrika")**

Bir LSTM katmanını, her zaman adımında (time step) neyi saklayacağına, neyi çöpe atacağına ve bir sonraki adıma neyi aktaracağına karar veren bir "mini fabrika" gibi düşünebilirsiniz.

Bu süreci üç küçük karar verici (**Gates/Kapılar**) ve **Cell State (Hücre Durumu)** adı verilen uzun vadeli bir taşıma bandı yönetir.



### 1. Forget Gate – “What can I safely ignore?”
**(Unutma Kapısı – "Neyi güvenle görmezden gelebilirim?")**

* **Job (Görevi):** Önceki zaman adımlarından gelen hangi bilginin artık gereksiz olduğuna karar verir ve onu siler.
* **How (Nasıl):** Son gizli duruma ($h_{t-1}$) ve mevcut girdiye ($x_t$) bakar. Sigmoid fonksiyonu kullanarak 0 ile 1 arasında bir sayı üretir.
    * `0` → Kesinlikle unut (throw it away).
    * `1` → Kesinlikle sakla (keep it).
* **Retail Example:** Model, Şubat ayı satışlarını tahmin ederken geçen yılki "Black Friday" (Efsane Cuma) sıçramasının artık bir gürültü (noise) olduğuna karar verir. Bu bilgiye `0`'a yakın bir değer atar ve hafızadan siler.

### 2. Input Gate – “What new info is worth storing?”
**(Girdi Kapısı – "Hangi yeni bilgi saklamaya değer?")**

* **Job (Görevi):** Hangi yeni bilginin ağın hafızasına (Cell State) ekleneceğini belirler.
* **How (Nasıl):** İki aşamalı çalışır:
    1.  **Sigmoid Filtresi (Sarı):** Hangi değerlerin güncelleneceğine karar verir (0-1 arası önem derecesi).
    2.  **Tanh Aday Katmanı (Pembe):** Hafızaya eklenebilecek yeni değer vektörünü (adayları) oluşturur (-1 ile 1 arası).
    * Bu ikisinin çarpımı hafızaya eklenir.
* **Retail Example:** Model ani bir "3 Günlük İndirim" kampanyası görür. Bunun önemli olduğuna karar verir (Sigmoid $\approx$ 1) ve kampanya etkisini taşıma bandına (belt) yazar.

### 3. Cell State – “Long-term memory lane”
**(Hücre Durumu – "Uzun vadeli hafıza şeridi")**

* **Job (Görevi):** LSTM'in "gerçek" hafızasıdır. Bilgiyi çok uzun süreler boyunca bozulmadan taşımasını sağlar.
* **How (Nasıl):** Hücrenin üzerinden dümdüz akan bir taşıma bandı (conveyor belt) gibidir. Sadece "Unutma" ve "Ekleme" adımlarıyla üzerinde küçük değişiklikler yapılır. Matematiksel işlemler lineer olduğu için (çarpma yerine toplama ağırlıklı), gradyanlar kaybolmadan geriye akabilir.
* **In Retail:** Mevsimsellik bilgisini veya genel trendi (trend), yüzlerce gün boyunca solmadan (without fading) taşıyan yapıdır.

### 4. Output Gate – “What should I reveal right now?”
**(Çıktı Kapısı – "Şu an neyi açığa çıkarmalıyım?")**

* **Job (Görevi):** Mevcut hafızaya dayanarak, şu anki zaman adımında (t) ne çıktı verileceğini seçer.
* **How (Nasıl):** Güncellenmiş hücre durumunu (Cell State) alır, bir `tanh` işleminden geçirir ve bunu yeni bir `sigmoid` filtresiyle çarparak bugünün gizli çıktısını ($h_t$) oluşturur. Bu çıktı:
    1.  Bir sonraki LSTM ünitesine ($t+1$) gider.
    2.  Gerçek tahmini yapan yoğun katmana (Dense Layer) gider.

#### 🏪 Retail Scenario: Output Gate Logic
**(Perakende Senaryosu: Çıktı Kapısı Mantığı)**

Bir LSTM'in hafızasında (Cell State) halihazırda mevsimsellik ("Hafta sonları çok satar") ve promosyon ("%20 kupon satışı artırır") bilgisinin saklı olduğunu hayal edin. Takvim **15 Mart Salı**'yı gösterdiğinde, Çıktı Kapısı hafıza vektörünün her parçası için şu soruları sorar:

| Memory Component <br> (Hafıza Bileşeni) | Forget Gate Decision <br> (Unutma Kapısı Durumu) | Output Gate Decision (Today) <br> (Çıktı Kapısı Kararı - Bugün) | Why? <br> (Neden?) |
| :--- | :--- | :--- | :--- |
| **Weekend Boost** <br> *(Hafta Sonu Etkisi)* | Kept at 100% <br> *(Cuma geliyor, sakla)* | **0.1 → Reveal only 10%** <br> *(Sadece %10'unu göster)* | Bugün Salı, hafta sonu bilgisi bugünkü satış tahmini için henüz yararlı değil. |
| **Coupon-Promo Effect** <br> *(Kupon İndirim Etkisi)* | Kept at 60% <br> *(Kupon hala geçerli)* | **0.8 → Reveal most of it** <br> *(Çoğunu göster)* | Kupon Çarşamba bitiyor, yani bugün talebi etkilemeli. |
| **Christmas Peak** <br> *(Noel Zirvesi)* | Kept at 100% <br> *(Uzun vadeli hafıza)* | **0.0 → Reveal nothing** <br> *(Hiçbir şey gösterme)* | Mart ayındayız; Noel bilgisinin bugünkü tahmini şişirmesine izin verme. |

---

## 🧠 Conceptual Flow: The Conveyor Belt
**(Kavramsal Akış: Taşıma Bandı)**

Aşağıdaki şema, bilginin LSTM hücresi içindeki akışını özetler:

<img width="700" height="350" alt="image" src="https://github.com/user-attachments/assets/e8dccd36-58c8-4d4c-b3fd-8a4c0d62b97b" />


## 🎯 Expert Summary: The Power of Gates
**(Uzman Özeti: Kapıların Gücü)**

> **Core Logic:** Her kapı (gate) eğitim sırasında kendi ağırlıklarını (weights) öğrenir. Bu dinamik yapı sayesinde LSTM, çelişkili gibi görünen görevleri **tek bir model içinde** (all in one model) başarıyla yönetir:
>
> 1.  **Forget:** Haftalık gürültüyü ve gereksiz veriyi unutur (Forgetting weekly noise).
> 2.  **Remember:** Mevsimsel döngüleri ve uzun vadeli trendleri hatırlar (Remembering seasonal cycles).
> 3.  **React:** Ani gelişen olaylara ve şoklara tepki verir (Reacting to sudden events).

---

## 🏆 Powerhouse Use Cases
**(Güç Merkezi Kullanım Alanları)**

LSTM, kısa vadeli oynaklık (volatility) ile uzun vadeli trendlerin iç içe geçtiği alanlarda endüstri standardı bir "Güç Merkezi"dir:

* 📈 **Sales Forecasting (Satış Tahmini):**
    * *Short-term:* Promosyon kaynaklı ani sıçramalar (Promo spikes).
    * *Long-term:* Noel/Bayram sezonu etkileri (Seasonal effects).
* 💹 **Stock-Price Moves (Hisse Senedi Hareketleri):**
    * *Short-term:* Gün içi dalgalanmalar/gürültü (Intraday jitter).
    * *Long-term:* Makroekonomik trendler ve döngüler (Macroeconomic trends).
* 🌦 **Weather Prediction (Hava Durumu Tahmini):**
    * *Short-term:* Saatlik sıcaklık değişimleri ve ani yağışlar (Hourly fluctuations).
    * *Long-term:* Yıllık iklim döngüleri (Yearly climate cycles).
* ⚡ **Energy Consumption (Enerji Tüketimi):**
    * *Short-term:* Anlık yük değişimleri ve talep artışları (Instant load changes).
    * *Long-term:* Haftalık ve mevsimsel kullanım kalıpları (Weekly usage patterns).
 
    * 

# 🔄 Summary: End-to-End ML/DL Workflow for Time Series
**(Özet: Zaman Serileri için Uçtan Uca ML/DL İş Akışı)**

Zaman serisi problemleri, standart denetimli öğrenme (supervised learning) problemlerinden farklıdır. Veri noktaları bağımsız değildir (not i.i.d.); zamanın akışı, otokorelasyon ve sıralama kritiktir. Aşağıdaki iş akışı, modern bir veri bilimcisinin **Klasik ML** (XGBoost, LightGBM) ve **Derin Öğrenme** (RNN, LSTM, Transformer) yaklaşımlarını uygularken izlemesi gereken standart prosedürü tanımlar.

---

## 1. Problem Framing
**(Problemin Çerçevelenmesi)**

Başarılı bir model, kod yazmadan önce doğru tanımla başlar.
* **Pick the Granularity (Granülariteyi Seçin):** Veri sıklığını belirleyin (saatlik, günlük, haftalık...).
    * *Trade-off:* Veri çok seyrekse (aylık) sinyal azdır; çok sıksa (dakikalık) gürültü (noise) fazladır.
* **Decide the Forecast Horizon (Tahmin Ufkuna Karar Verin):** Ne kadar ileriye tahmin yapılacak? (önümüzdeki 24 saat, gelecek 12 hafta...).
    * *Strategy:* Kısa vade için "One-step ahead", uzun vade için "Multi-step direct" veya "Recursive" stratejiler seçilir.
* **Choose the Business Metric (İş Metriğini Seçin):** Optimizasyon hedefi iş ihtiyacına uymalıdır.
    * **MAE/MAPE:** Talep tahmini (Demand) için yaygındır (yorumlanabilirdir).
    * **RMSE:** Büyük hataları ağır cezalandırmak gerekiyorsa.
    * **Quantile Loss / Service Level:** Envanter yönetimi (Inventory) için (stoksuz kalmama veya aşırı stok maliyeti dengesi).

---

## 2. Feature Engineering
**(Özellik Mühendisliği)**

Veriyi modele nasıl sunduğunuz, algoritma türüne göre radikal biçimde değişir.

### A. For Classical ML Models (Trees, Linear, SVR)
Model "zamanı" ve "sırayı" bilmez, ona biz öğretmeliyiz.
* **Lag Columns (Gecikme Sütunları):** $t-1, t-7, t-30$ gibi geçmiş değerler. Otokorelasyonu yakalar.
* **Rolling Window Statistics (Kayan Pencere İstatistikleri):** Trend ve volatiliteyi yakalamak için kayan ortalamalar (rolling means) ve standart sapmalar (rolling stds).
* **Calendar Flags (Takvim İşaretçileri):** Mevsimselliği yakalamak için. Haftanın günü (weekday dummies), ay, tatil (holiday binary) bilgileri.
* **External Regressors (Dışsal Değişkenler):** Hava durumu, promosyon bayrağı, web trafiği.
* **Target Transforms (Hedef Dönüşümleri):**
    * Log veya Box-Cox dönüşümü, varyansı stabilize etmek için kullanılır.
    * *Critical Note:* Bu, Lineer Regresyon, SVR, kNN için şarttır (Gaussian varsayımı). Ağaç tabanlı (Tree-based) algoritmalar için zorunlu değildir ancak performansı artırabilir.

### B. For DL Models (RNN/LSTM/Transformers)
Derin öğrenme, ham dizilerden özellik çıkarabilir.
* **Raw History (Ham Geçmiş):** Genellikle Lag veya Rolling özelliklere manuel ihtiyaç yoktur; sıralı modeller (sequence models) ham geçmişi okuyarak bu kalıpları öğrenir.
* **Embeddings (Gömülü Öznitelikler):** Kategorik değişkenler (shop_id, item_id) için "One-Hot Encoding" yerine, öğrenilebilir vektörler olan Embedding katmanları kullanılır. Yüksek kardinalite (high cardinality) için kritiktir.
* **Scaling (Ölçeklendirme):**
    * Sürekli kanalları (continuous channels) normalleştirin/standartlaştırın (MinMax veya Z-Score). *DL modelleri ölçeklendirilmemiş veride yakınsamaz (converge).*
    * İkili (binary) değişkenleri 0/1 olarak bırakın.
* **Known-Future Features (Bilinen Gelecek Özellikleri):** Gelecek zaman adımları için bilinen veriler (fiyat takvimi, promosyon planı) modele "decoder" veya ek girdi olarak verilir.

---

## 3. Train / Validation Split
**(Eğitim / Doğrulama Ayrımı)**

Zaman serilerinde **Asla Karıştırma Yapılmaz (No Shuffling)!** Gelecek verisi geçmişe sızmamalıdır (Data Leakage).

### Time-Based Only Strategy
* **Classical ML (Expanding Window / Walk-Forward):**
    * **Expanding-Window Back-test:** İlk N gözlemle başla, eğit $\rightarrow$ sonraki bloğu test et. Sonra pencereyi genişlet, tekrar eğit ve test et. Gerçek hayat performansını en iyi simüle eden yöntemdir.
    * **Walk-Forward:** Her adımda modeli yeniden sığdırır (re-fit). Küçük veri setleri için iyidir ancak hesaplama maliyeti yüksektir.
* **Deep Learning (Early Stopping):**
    * **Early-Stopping Split:** Serinin son %10-20'sini bir "doğrulama bloğu" olarak ayırın. Eğitim hatası düşerken doğrulama hatası artmaya başladığında (overfitting) eğitimi durdurun.
    * **Rolling Validation:** Eğer kaynaklar elveriyorsa, DL modelleri için de çoklu yeniden eğitim (multiple re-trains) yapılabilir.
    * *Note:* Batch'lerin kronolojik sırayı takip ettiğinden emin olun (özellikle stateful RNN'ler için).

---

## 4. Model Selection & Tuning
**(Model Seçimi ve Ayarlama)**

### ML (Machine Learning)
* **Grid / Bayesian Search:** Ağaç derinliği (depth), öğrenme oranı (learning rate) gibi hiperparametreler için Optuna veya basit Grid Search kullanın.
* **Evaluation:** Her konfigürasyonu Adım 3'teki "Expanding-window" testi ile değerlendirin.
* **Selection Principle:** İş hedeflerini tutturan en düşük hataya (MAE/MAPE) sahip ve **en basit** modeli seçin (Occam's Razor).

### DL (Deep Learning)
* **Architecture Tuning:** Katman sayısı, gizli birimler (hidden units), Dropout oranı, Dikkat başlıkları (Attention heads).
* **Optimizer Schedule:** Öğrenme oranı (Learning Rate) en kritik parametredir. Batch size ve Epoch sayısı ayarlanmalıdır.
* **Metric:** Erken durdurma (early stopping) için doğrulama kaybını (validation loss) kullanın. Eğer birden fazla konfigürasyon yakınsarsa, onları ayırdığınız test setindeki (hold-out) MAE/MAPE'ye göre sıralayın.

---

## 5. Residual Diagnostics
**(Artık Değer Teşhisi)**

Model eğitildikten sonra hataları (residuals = Gerçek - Tahmin) analiz edin.
* **Check:** Artıklar hala bir desen (pattern) gösteriyor mu?
* **Autocorrelation:** Eğer artıklar arasında otokorelasyon varsa, model bazı sinyalleri kaçırmıştır. Daha fazla Lag ekleyin.
* **Seasonality:** Eğer hatalarda dönemsellik varsa, mevsimsel kukla değişkenler (seasonal dummies) ekleyin.

---

## 6. Deployment & Monitoring
**(Dağıtım ve İzleme)**

Model canlıya alındığında iş bitmez.
* **Automate Retraining:** Yeni veriler geldikçe modelin periyodik olarak yeniden eğitilmesini otomatize edin.
* **Track Drift:** Canlı hata oranını (live error drift) izleyin. Veri dağılımı değişti mi? (Concept Drift).
* **Alarms:** Rejim değişiklikleri (regime changes) veya beklenmedik anormallikler için alarmlar kurun.

---

### 📌 Expert Advice (Uzman Tavsiyesi)

> Eğer zaman serileriniz **kısa, temiz ve düşük gürültülü** (short, clean, low-noise) ise, karmaşık modellere girmeden klasik **ARIMA/SARIMA** ile başlayın.
>
> Ancak veri karmaşıksa (yüksek gürültü, çoklu dışsal değişkenler), bir ML yaklaşımı —önce **Boosted Trees (XGBoost/LightGBM)** ile başlayıp, gerekirse **Deep Learning (LSTM/TFT)** tarafına geçmek— genellikle daha düşük hata oranları ve daha zengin içgörüler (richer insights) sağlar.
