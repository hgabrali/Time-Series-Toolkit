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
