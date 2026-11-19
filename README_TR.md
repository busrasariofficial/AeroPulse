
# AeroPulse ✈️
### ABD Uçuş Gecikmesi Tahmini ve Operasyonel Analiz

**Dil:** 🇹🇷 Türkçe | [🇬🇧 English](README.md)

Python, istatistiksel analiz, makine öğrenmesi, açıklanabilir yapay zekâ ve zamansal doğrulama yöntemlerini bir araya getiren uçtan uca bir veri bilimi projesi.

**Teknolojiler:** Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn

---

## Proje Hakkında

Uçuş gecikmeleri; havayolu şirketleri, havalimanları ve yolcular açısından operasyonel sorunlara neden olmaktadır.

AeroPulse projesi, ABD'deki uçuşların kalkış gecikmelerini geçmiş uçuş verileri ve planlanan uçuş özellikleri üzerinden tahmin etmeyi amaçlamaktadır.

Proje yalnızca bir tahmin modeli geliştirmekle sınırlı değildir. Aynı zamanda modelin güvenilirliğini, olasılık kalibrasyonunu, karar eşiklerini ve farklı operasyonel koşullardaki performansını incelemektedir.

### Araştırma Soruları

- Planlanan uçuş bilgileri kullanılarak 15 dakika ve üzerindeki kalkış gecikmeleri tahmin edilebilir mi?
- Logistic Regression, Random Forest ve XGBoost modellerinden hangisi daha başarılıdır?
- Gecikme tahminlerini en fazla hangi değişkenler etkilemektedir?
- Olasılık kalibrasyonu tahminlerin güvenilirliğini artırabilir mi?
- Ağustos verileriyle geliştirilen model Eylül ayında da başarılı olabilir mi?
- Model performansı havayolu, havalimanı ve kalkış saatine göre nasıl değişmektedir?

---

## Veri Seti

**Kaynak:** ABD Ulaştırma Bakanlığı, Bureau of Transportation Statistics (BTS)

**Veri seti:** Airline On-Time Performance Data

**Dönem:** Ağustos–Eylül 2026

| Veri Seti | Uçuş Sayısı | Kullanım Amacı |
|---|---:|---|
| Ağustos 2026 | 594.207 | Model geliştirme ve iç değerlendirme |
| Eylül 2026 | 563.591 | Dış dönem doğrulaması |
| **Toplam** | **1.157.798** | **Analiz edilen tamamlanmış uçuşlar** |

İptal edilen ve başka havalimanına yönlendirilen uçuşlar modelleme veri setinin dışında bırakılmıştır.

### Hedef Değişken

`Delayed15`

- `0`: Kalkış gecikmesi 15 dakikadan az
- `1`: Kalkış gecikmesi 15 dakika veya daha fazla

### Kullanılan Değişkenler

**Zamansal değişkenler**
- Haftanın günü
- Hafta sonu göstergesi
- Planlanan kalkış saati
- Kalkış zaman dilimi
- Döngüsel saat dönüşümleri (`HourSin`, `HourCos`)

**Uçuş özellikleri**
- Havayolu şirketi
- Kalkış havalimanı
- Varış havalimanı
- Logaritmik dönüşüm uygulanmış uçuş mesafesi

Tahmin değişkenleri arasında gerçekleşmiş kalkış veya varış gecikmesi kullanılmamıştır.

---

## Metodoloji

### 1. Veri Hazırlama

- Ham BTS verilerinin incelenmesi
- Veri temizleme
- Eksik değer analizleri
- Uçuş durumlarının değerlendirilmesi
- Özellik mühendisliği

### 2. Keşifsel ve İstatistiksel Analiz

- Gecikme dağılımları
- Havayolu bazında gecikme oranları
- Havalimanı istatistikleri
- Saatlik gecikme örüntüleri
- Uçuş mesafesi analizleri
- İstatistiksel karşılaştırmalar

### 3. Makine Öğrenmesi

Üç sınıflandırma yaklaşımı incelenmiştir:

- Logistic Regression
- Random Forest
- XGBoost

Sonraki kalibrasyon ve dış dönem doğrulama aşamalarında XGBoost kullanılmıştır.

### 4. Açıklanabilir Yapay Zekâ

Modelin tahmin davranışını anlamak amacıyla:

- Özellik önem analizi
- SHAP analizi
- Ablation Study (değişken grubu çıkarma analizi)

uygulanmıştır.

### 5. Olasılık Kalibrasyonu

Üç yaklaşım karşılaştırılmıştır:

- Orijinal XGBoost olasılıkları
- Sigmoid kalibrasyonu
- Isotonic kalibrasyonu

Son değerlendirme sürecinde sigmoid kalibrasyonu tercih edilmiştir.

### 6. Karar Eşiği Optimizasyonu

Ağustos doğrulama verisi üzerinden bir sınıflandırma eşiği seçilmiştir.

**Nihai karar eşiği: `0.253331`**

Bu eşik Eylül verilerine yeniden optimize edilmeden uygulanmıştır.

### 7. Dış Dönem Doğrulaması

Ağustos 2026 verileriyle geliştirilen model, Eylül 2026 verileri üzerinde test edilmiştir.

Böylece yalnızca rastgele bir eğitim-test ayrımına bağlı kalmadan zamansal genelleme performansı değerlendirilmiştir.

### 8. Sağlamlık ve Hata Analizi

Model performansı şu kırılımlarda incelenmiştir:

- Havayolu şirketleri
- Kalkış zaman dilimleri
- Planlanan kalkış saatleri
- Kalkış havalimanları
- Varış havalimanları
- Uçuş tarihleri
- Tahmin edilen risk aralıkları

---

## Zamansal Veri Ayrımı

| Veri Bölümü | Tarih | Amaç |
|---|---|---|
| Eğitim | 1–20 Ağustos 2026 | Model eğitimi |
| Doğrulama | 21–25 Ağustos 2026 | Kalibrasyon ve eşik seçimi |
| İç test | 26–31 Ağustos 2026 | İç performans değerlendirmesi |
| Dış doğrulama | 1–30 Eylül 2026 | Zamansal genelleme testi |

Ağustos doğrulama verisi hem kalibrasyon hem de eşik optimizasyonunda kullanılmıştır. Bu nedenle doğrulama performansı tamamen bağımsız bir değerlendirme değildir.

Eylül verileri dış dönem değerlendirmesi olarak kullanılmıştır.

---

## Nihai Model Performansı

### Eylül 2026 — Dış Dönem Testi

Model **563.591 tamamlanmış uçuş** üzerinde değerlendirilmiştir.

| Performans Metriği | Sonuç |
|---|---:|
| ROC-AUC | **0,7142** |
| PR-AUC | **0,4249** |
| F1-Score | **0,4588** |
| Precision (Kesinlik) | 0,3988 |
| Recall (Duyarlılık) | 0,5401 |
| Accuracy (Doğruluk) | 0,7094 |
| Brier Score | **0,1577** |
| Log Loss | 0,4861 |
| Karar Eşiği | 0,253331 |

Eylül ayında gerçekleşen kalkış gecikmesi oranı **%22,81** olarak hesaplanmıştır.

### Karmaşıklık Matrisi (Confusion Matrix)

| | Zamanında Tahmin | Gecikmeli Tahmin |
|---|---:|---:|
| Gerçekte Zamanında | 330.366 | 104.667 |
| Gerçekte Gecikmeli | 59.120 | 69.438 |

Seçilen karar eşiğiyle model, geciken uçuşların yaklaşık **%54'ünü** tespit etmiştir.

Bununla birlikte yanlış gecikme uyarılarının sayısı da yüksektir.

Bu durum, gecikmeleri yakalama oranı ile yanlış alarm oranı arasındaki dengeyi göstermektedir.

---

## Önemli Bulgular

### 1. Kalkış Saati Önemli Bir Tahmin Değişkenidir

XGBoost özellik önem analizinde zamansal değişkenler öne çıkmıştır.

- `HourSin`: yaklaşık %45,24
- `DepPeriod`: yaklaşık %27,61

Bu değerler modelin değişkenleri nasıl kullandığını göstermektedir; doğrudan nedensel etki olarak yorumlanmamalıdır.

### 2. Gecikmeler Kalkış Zamanına Göre Değişmektedir

| Kalkış Dönemi | Gerçek Gecikme Oranı | Model Recall |
|---|---:|---:|
| Sabah | %12,95 | %10,50 |
| Öğleden Sonra | %27,37 | %56,15 |
| Akşam | %34,62 | %80,48 |
| Gece | %22,23 | %57,12 |

Akşam uçuşlarında gecikme oranı en yüksek seviyededir.

Model, seçilen karar eşiğinde akşam gecikmelerini sabah gecikmelerine kıyasla daha yüksek oranda yakalamaktadır.

### 3. Havayolları Arasında Performans Farkları Vardır

| Havayolu | Gecikme Oranı | Recall |
|---|---:|---:|
| Southwest (WN) | %31,64 | %75,02 |
| Delta (DL) | %16,94 | %30,33 |
| American (AA) | %26,97 | %56,50 |
| United (UA) | %20,04 | %46,66 |
| JetBlue (B6) | %25,80 | %76,43 |

Bu sonuçlar modelin yalnızca genel metriklerle değil, alt gruplar bazında da değerlendirilmesi gerektiğini göstermektedir.

### 4. Kalibrasyon Analizi

Kalibre edilmiş modelin Eylül Brier Score değeri **0,1577** olarak hesaplanmıştır.

Risk aralığı analizlerinde modelin bazı gruplarda gerçekleşen gecikme oranından biraz daha düşük olasılık tahmin ettiği gözlemlenmiştir.

Bu nedenle kalibrasyonun başarılı olması, bütün alt gruplarda kusursuz olasılık tahmini yapıldığı anlamına gelmemektedir.

### 5. Zamansal Doğrulamanın Önemi

Modelin Ağustos ayında geliştirilip Eylül ayında değerlendirilmesi, kısa dönemli zamansal genelleme açısından ek kanıt sağlamaktadır.

Bununla birlikte farklı mevsimler ve yıllar üzerindeki performansı henüz test edilmemiştir.

---

## Görselleştirmeler

### Kalkış Dönemine Göre Model Performansı

![Kalkış Dönemi F1](outputs/robustness/departure_period_f1.png)

### Havayolu Bazında Recall

![Havayolu Recall](outputs/robustness/airline_recall.png)

### Saatlik Kaçırılan Gecikmeler

![Saatlik Miss Rate](outputs/robustness/hourly_miss_rate.png)

### Havalimanı Bazında Yanlış Alarm Analizi

![Havalimanı False Alarm](outputs/robustness/origin_false_alarm_rate.png)

### Olasılık Kalibrasyonu

![Eylül Risk Kalibrasyonu](outputs/robustness/september_risk_calibration.png)

---

## Model Bütünlüğü ve Tekrarlanabilirlik

Model bütünlüğünü değerlendirmek amacıyla ayrı bir teknik denetim gerçekleştirilmiştir.

Denetim kapsamında:

- Dosya ve model özelliklerinin tutarlılığı
- Eksik değer kontrolleri
- Zamansal veri ayrımı
- Özellik mühendisliği doğrulaması
- Kategori değişimleri
- Tahminlerin yeniden üretilebilirliği
- Karar eşiği tutarlılığı
- Dosya bütünlüğü

incelenmiştir.

### Denetim Sonuçları

| Durum | Kontrol Sayısı |
|---|---:|
| PASS | 22 |
| FAIL | 0 |
| REVIEW | 3 |

Eylül tahminleri yeniden üretildiğinde kaydedilmiş tahminlerle maksimum fark **0,0000000000** olarak bulunmuştur.

Orijinal model eğitimi ve kalibrasyon kodları da ayrıca incelenmiştir.

Otomatik denetimdeki üç inceleme kaydı, şeffaflık amacıyla korunmuştur.

---

## Proje Klasör Yapısı

```text
AeroPulse/
├── src/
│   ├── 01_initial_eda.py
│   ├── 02_data_cleaning.py
│   ├── 03_exploratory_analysis.py
│   ├── ...
│   ├── 15_model_robustness.py
│   ├── 16_model_integrity.py
│   └── 17_final_report.py
├── data/
│   └── processed/
├── outputs/
│   ├── calibration/
│   ├── explainability/
│   ├── external_validation/
│   ├── figures/
│   ├── final_report/
│   ├── integrity/
│   ├── models/
│   ├── robustness/
│   ├── statistics/
│   ├── tables/
│   └── threshold_validation/
├── README.md
├── README_TR.md
├── requirements.txt
└── .gitignore
```

Büyük veri setleri ve kaydedilmiş model dosyaları GitHub deposuna dahil edilmemektedir.

---

## Kurulum

Projeyi bilgisayarına klonla:

```bash
git clone https://github.com/busrasariofficial/AeroPulse.git
cd AeroPulse
```

Sanal ortam oluştur:

```bash
python -m venv venv
```

Windows üzerinde sanal ortamı etkinleştir:

```bash
venv\Scripts\activate
```

Bağımlılıkları yükle:

```bash
pip install -r requirements.txt
```

Orijinal BTS veri setlerinin ayrıca indirilmesi ve uygun veri klasörlerine yerleştirilmesi gerekmektedir.

Analizlerin yeniden çalıştırılabilmesi için ara dosyaların da doğru sırayla üretilmesi gerekir.

---

## Projenin Sınırlamaları

- Çalışma yalnızca iki ardışık ayı kapsamaktadır.
- Kalibrasyon ve eşik seçimi aynı doğrulama dönemini kullanmaktadır.
- Sınıf dengesizliği performans metriklerini etkilemektedir.
- Havayolu ve havalimanı bazında performans farklılıkları bulunmaktadır.
- Bazı Eylül havalimanı kategorileri Ağustos referans verisinde bulunmamaktadır.
- Gerçek zamanlı hava durumu ve hava trafik yoğunluğu verileri kullanılmamıştır.
- Model üretim ortamına hazır bir sistem değil, araştırma ve portföy prototipidir.
- Bulgular nedensellik kanıtı olarak yorumlanmamalıdır.

---

## Gelecekte Yapılabilecek Geliştirmeler

- Çok aylı ve çok yıllı zamansal doğrulama
- Hava durumu ve havalimanı yoğunluğu değişkenlerinin eklenmesi
- Alt gruplar için daha ayrıntılı kalibrasyon
- Model izleme ve veri kayması tespiti
- Tahmin servisinin geliştirilmesi

Bu geliştirmeler mevcut projenin kapsamı dışındadır.

---

## Teknik Raporlar

- [İngilizce Final Teknik Rapor](outputs/final_report/AeroPulse_Final_Report.md)
- [Yönetici Özeti](outputs/final_report/AeroPulse_Executive_Summary.md)
- [Doğrulanmış Eylül Metrikleri](outputs/final_report/final_metrics.csv)

---

## Geliştirici

**Büşra Sarı**

İstatistik Mezunu | Veri Analitiği ve Makine Öğrenmesi

[GitHub Profilim](https://github.com/busrasariofficial)

---

*Bağımsız bir veri bilimi portföy projesi olarak geliştirilmiştir.*
