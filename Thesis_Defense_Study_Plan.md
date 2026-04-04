# Tez Savunma Hazırlık Planı
# Uncertainty-Aware Bus ETA Prediction Under Drift with Segment-Level Risk Decomposition

**Amaç**: Tezin her detayına hakim olarak savunmaya hazırlanmak
**Yöntem**: Her modülde önce ön bilgi kontrolü → eksik varsa tamamla → konuyu öğret → anlama kontrolü
**Dil**: Türkçe (teknik terimler İngilizce kalacak)

---

## MODÜL 0: Temel İstatistik ve ML Temelleri (Ön Koşul)
> *Bu modül, sonraki tüm konuların üzerine inşa edileceği temeldir.*

### 0.1 Olasılık ve İstatistik Temelleri
- [ ] Probability distribution nedir? (PDF, CDF kavramları)
- [ ] Quantile (yüzdelik dilim) kavramı ve hesaplanması
- [ ] Empirical distribution vs theoretical distribution farkı
- [ ] Hypothesis testing: p-value, significance level (α)
- [ ] Non-parametric testler: Kolmogorov-Smirnov testi, Kruskal-Wallis testi
- [ ] IQR (Interquartile Range) ve outlier tespiti

### 0.2 Machine Learning Temelleri
- [ ] Supervised learning: regression vs classification
- [ ] Overfitting vs underfitting, bias-variance trade-off
- [ ] Cross-validation: neden temporal split gerekir?
- [ ] Feature engineering: neden önemli? Data leakage nedir?
- [ ] Evaluation metrics: MAE, RMSE, MAPE — ne zaman hangisi?

### 0.3 Decision Trees ve Ensemble Yöntemleri
- [ ] Decision tree nasıl çalışır? (split criteria, leaf nodes)
- [ ] Bagging vs Boosting farkı
- [ ] Gradient Boosting kavramı
- [ ] XGBoost: neden tercih edildi? Avantajları neler?
- [ ] Hyperparameter'lar: n_estimators, max_depth, learning_rate, subsample, reg_lambda

---

## MODÜL 1: Problem Tanımı ve Motivasyon
> *"Bu tez neden önemli?" sorusuna eksiksiz cevap verebilmek*

### 1.1 Bus ETA Prediction Problemi
- [ ] ETA prediction nedir ve neden önemli?
- [ ] Point prediction vs prediction interval farkı
- [ ] Neden sadece nokta tahmini yeterli değil?
- [ ] Operasyonel karar verme için belirsizlik bilgisi neden kritik?

### 1.2 Temporal Distribution Shift (Drift)
- [ ] Distribution shift / concept drift nedir?
- [ ] Covariate shift vs concept drift vs prior probability shift ayrımları
- [ ] Trafik verilerinde drift neden kaçınılmaz?
- [ ] Stationarity ve non-stationarity kavramları
- [ ] Veri setindeki drift kanıtları (KS testi, Kruskal-Wallis sonuçları)

### 1.3 Research Gap ve Tez Katkısı
- [ ] Mevcut literatürün eksiklikleri neler?
- [ ] Bayesian yöntemlerin sınırlamaları (MCMC computational cost)
- [ ] Kalman filter'ın Gaussian assumption sınırlaması
- [ ] Bu tezin 4 ana katkısı neler?

---

## MODÜL 2: Veri Seti ve Preprocessing Pipeline
> *"Veriniz hakkında ne biliyorsunuz?" sorusuna detaylı cevap*

### 2.1 GTFS Veri Formatı
- [ ] GTFS (General Transit Feed Specification) nedir?
- [ ] trips, stops, stop_times, routes, calendar_dates tabloları
- [ ] Segment-level vs route-level veri yapısı
- [ ] Bu veri setinin seçim gerekçesi (literatüre geçmiş, segment yapısı mevcut)

### 2.2 Veri Seti Özellikleri
- [ ] 785,976 kayıt, 55 gün, 3 rota, 201 durak, 19,769 trip
- [ ] Astana (Kazakistan) şehir içi otobüs verileri
- [ ] run_time ve dwell_time kavramları
- [ ] total_segment_time = run_time + dwell_time

### 2.3 Veri Temizleme Adımları
- [ ] Duplicate removal: neden ve nasıl?
- [ ] Outlier detection: IQR-based per-segment-direction — neden gruplu?
- [ ] 925 zero-value run time'ın kaldırılması
- [ ] Incomplete trip filtering (< 30 segment)
- [ ] Anomalous dates (Sep 3-4): neden ve nasıl tespit edildi?

### 2.4 Temporal Split Stratejisi
- [ ] Neden random split değil temporal split?
- [ ] W1-W3 (train) → W4 (cal) → W5 (near) → W6 (mid) → W7-W8 (far)
- [ ] Her dönemin amacı: artan temporal distance ile drift etkisini ölçmek
- [ ] Data leakage prevention: neden geleceğe ait veri kullanılamaz?

---

## MODÜL 3: Feature Engineering
> *"Hangi feature'ları kullandınız ve neden?" sorusuna tam cevap*

### 3.1 Temporal Features (8 adet)
- [ ] hour_of_day, day_of_week, is_weekend, time_period
- [ ] Cyclical encoding: sin/cos dönüşümü — neden gerekli?
  - `hour_sin = sin(2π × hour / 24)`, `hour_cos = cos(2π × hour / 24)`
- [ ] time_period sınıflandırması (6 dönem): neden yapıldı?

### 3.2 Historical Statistics (12 adet: 6 route + 6 segment)
- [ ] 7-day rolling window: mean, std, median, Q25, Q75, count
- [ ] Past-only lookback: neden sadece geçmiş veriden hesaplanıyor?
- [ ] Per (segment/route, direction, time_period) gruplama mantığı
- [ ] Fallback mekanizması: yetersiz geçmiş veri durumunda global istatistikler

### 3.3 Route Context Features (2-6 adet)
- [ ] route_id_encoded, direction_encoded
- [ ] segment_number_normalized, total_route_segments

### 3.4 Trip Progress Features (6 adet, sadece segment-level)
- [ ] cumulative_time_so_far, segments_completed, fraction_route_completed
- [ ] prev_seg_run_time, prev_seg_dwell_time, prev_2/3_seg_avg
- [ ] Neden sadece segment-level modelde var?

### 3.5 Feature Importance Analizi
- [ ] hist_route_mean (%27) neden baskın?
- [ ] Historical features (%56) dominansının anlamı: model geçmişe bağımlı → drift'e açık
- [ ] Bu bulgunun conformal prediction ihtiyacını nasıl desteklediği

---

## MODÜL 4: Conformal Prediction — Teorik Temeller ★★★
> *Tezin en kritik teorik bileşeni. Savunmada en çok soru gelecek alan.*

### 4.1 Uncertainty Quantification Yaklaşımları
- [ ] Parametrik yöntemler (Gaussian assumption, Student-t)
- [ ] Bayesian yöntemler (posterior distribution, MCMC)
- [ ] Bootstrap yöntemleri
- [ ] **Conformal Prediction**: distribution-free, finite-sample guarantees

### 4.2 Split Conformal Prediction — Temel Algoritma
- [ ] Exchangeability assumption: nedir ve neden önemli?
- [ ] Nonconformity score: `R_i = |y_i - ŷ_i|` (residual)
- [ ] Quantile hesaplama: `q = Quantile(R, ⌈(n+1)(1-α)⌉ / n)`
  - n: kalibrasyon seti boyutu, α: hata oranı (ör. 0.10 → %90 coverage)
- [ ] Prediction interval: `[ŷ - q, ŷ + q]` (symmetric)
- [ ] **Coverage guarantee**: P(y ∈ C(x)) ≥ 1 - α (exchangeability altında)
- [ ] Neden "distribution-free"? Hiçbir dağılım varsayımı yok!
- [ ] Finite-sample guarantee vs asymptotic guarantee farkı

### 4.3 Coverage Guarantee'nin Matematik İspatı
- [ ] Exchangeability tanımı: P(Z_π(1), ..., Z_π(n+1)) = P(Z_1, ..., Z_n+1) ∀ permütasyon π
- [ ] Nonconformity score'un rank'ı uniform dağılır → quantile hesabı kesin garanti verir
- [ ] Bu garantinin KIRILMA koşulu: exchangeability ihlali = distribution shift!

### 4.4 Neden Statik CP Drift Altında Başarısız?
- [ ] Train/Cal dönemi ile test dönemi arasında dağılım değişirse exchangeability bozulur
- [ ] Kalibrasyon setindeki residual dağılımı artık test setini temsil etmez
- [ ] Sabit q değeri tüm tahminlere aynı genişlikte interval verir → uyumsuz
- [ ] Sonuç: PICP 0.90 → 0.61 (29 pp düşüş)

### 4.5 Online/Adaptive Conformal Prediction
- [ ] Temel fikir: kalibrasyon setini zamanda ilerledikçe güncelle
- [ ] **Expanding window**: her yeni gözlemi kalibrasyon setine ekle
  - Avantaj: daha fazla veri, daha stabil quantile
  - Dezavantaj: eski veriler ağırlık taşımaya devam eder
- [ ] **Sliding window**: sabit pencere boyutu, eski veriyi at
  - 7 gün: en dar interval ama en az coverage iyileştirmesi
  - 14 gün: orta yol
- [ ] Güncelleme frekansı: daily vs hourly
- [ ] Online CP'nin coverage-width trade-off'u

### 4.6 Normalized Conformal Prediction
- [ ] Problem: farklı segmentlerin farklı zorluk seviyeleri var
- [ ] MAD (Median Absolute Deviation) ile difficulty estimation
- [ ] Normalized score: `R_i / σ_s` (σ_s = segment-specific MAD)
- [ ] Adaptive interval: `[ŷ - q × σ_s, ŷ + q �� σ_s]`
- [ ] Zor segmentler → geniş interval, kolay segmentler → dar interval

### 4.7 Calibrated Explanations Framework
- [ ] WrapCalibratedExplainer: XGBoost'u conformal prediction ile sarmalıyor
- [ ] crepes kütüphanesi ile ilişki
- [ ] Conformal predictive distributions: P(y < t) sorgulama yeteneği
- [ ] Framework'ün implementasyondaki kullanımı

---

## MODÜL 5: Experiment 1 — Static CP Under Drift
> *"İlk deneyin tasarımını ve sonuçlarını açıklayın"*

### 5.1 Deney Tasarımı
- [ ] Amaç: temporal drift'in statik CP'ye etkisini ölçmek
- [ ] Kalibrasyon: W4 (2,740 trip), sabit
- [ ] Test dönemleri: W5 (near), W6 (mid), W7-W8 (far) — artan temporal distance
- [ ] Kontrol: kalibrasyon seti üzerinde coverage = 0.9007 (doğru implementasyon kanıtı)

### 5.2 Sonuçlar ve Yorumlama
- [ ] PICP: 0.90 → 0.62 (near) → 0.60 (mid) → 0.61 (far)
- [ ] MPIW: sabit 1,528s — neden sabit? (statik CP'nin doğası)
- [ ] Calibration error: 0.0007 → 0.283 → 0.297 → 0.293
- [ ] **Ani çöküş**: kademeli drift değil, rejim değişikliği
- [ ] Test-Near → Test-Far arasında stabil (ek bozulma yok)

### 5.3 Multi-Confidence ve Conditional Analiz
- [ ] %80, %90, %95 confidence seviyelerinde karşılaştırma
- [ ] Yüksek confidence → daha az coverage kaybı ama daha geniş interval
- [ ] Zaman dilimine göre conditional coverage: sabah vs öğlen vs gece
- [ ] Night sample size problemi ve yorumu

### 5.4 RQ1'e Cevap (Ezberle)
- [ ] "Temporal distribution shift, statik conformal prediction'ın empirical coverage'ını ciddi şekilde bozar. Coverage %90'dan %61'e düşer (29 pp). Bu düşüş kademeli değil, anidir — kalibrasyon dönemi sonrasında hemen gerçekleşir. Statik CP'nin sabit genişlikli interval'leri (1,528s) tahmin zorluğuna adapte olamaz."

---

## MODÜL 6: Experiment 2 — Online Adaptive CP
> *"Adaptive yaklaşımınız ne kadar başarılı?"*

### 6.1 Deney Tasarımı
- [ ] 3 varyant: Expanding, Sliding-7d, Sliding-14d
- [ ] Güncelleme mekanizması: her gün/saat yeni gözlemlerle kalibrasyon seti güncellenir
- [ ] Quantile her güncelleme sonrası yeniden hesaplanır

### 6.2 Sonuçlar
- [ ] Expanding: PICP 0.746, MPIW 2,164s (+42% daha geniş)
- [ ] Sliding-14d: PICP 0.721, MPIW 2,018s
- [ ] Sliding-7d: PICP 0.655, MPIW 1,626s (en verimli)
- [ ] Coverage stability: std dev 0.087 → 0.063 (expanding)

### 6.3 Hourly vs Daily (Exp2b)
- [ ] Hourly: +0.8 pp ek iyileşme, 24× hesaplama maliyeti
- [ ] Sonuç: daily güncelleme optimal trade-off
- [ ] Neden hourly çok az fark yarattı? (günlük trafik patternleri daha belirleyici)

### 6.4 Coverage-Width Trade-off
- [ ] Daha iyi coverage = daha geniş interval (kaçınılmaz trade-off)
- [ ] Winkler score: hem coverage hem width'i tek metrikte birleştirir
- [ ] Hiçbir online yöntem %90 hedefine ulaşamadı → drift çok ciddi

### 6.5 RQ2'ye Cevap (Ezberle)
- [ ] "Online conformal prediction, temporal drift'in neden olduğu miscalibration'ı kısmen düzeltir. Expanding window en iyi coverage'ı sağlar (0.746, +13.7 pp), ancak %42 daha geniş interval'ler gerektirir. %90 hedefine hiçbir yöntem ulaşamaz — bu, dağılım kaymasının salt kalibrasyon güncellemeleriyle çözülemeyecek kadar ciddi olduğunu gösterir. Daily güncelleme optimal trade-off'u sunar."

---

## MODÜL 7: Experiment 3 — Segment-Level Decomposition
> *"Segment analizi ne sağlıyor?"*

### 7.1 Normalized CP ve Segment Modeli
- [ ] Segment-level XGBoost (26 feature, 1,000 tree)
- [ ] Per-segment difficulty estimation via MAD
- [ ] Normalized nonconformity scores
- [ ] Her segmente özel genişlikte interval

### 7.2 Route-Level Aggregation
- [ ] Sum aggregation: route interval = [Σ lower_i, Σ upper_i]
- [ ] Neden toplama çalışıyor? (independence assumption)
- [ ] Bonferroni correction: çoklu test düzeltmesi (çok muhafazakar)
- [ ] Aggregated PICP: 0.983 (hedef %90'ın üzerinde)

### 7.3 Spatial Uncertainty Attribution
- [ ] Width ratio: en belirsiz / en belirli segment = 4.3×
- [ ] Top-5 segment: toplam belirsizliğin %22'si
- [ ] Directional asymmetry: Segment 1'de 11× fark (335s vs 30s MAE)
- [ ] Hangi segmentlerin en riskli olduğunu belirleme yeteneği

### 7.4 RQ3'e Cevap (Ezberle)
- [ ] "Segment-seviyesinde belirsizlik ayrıştırması, rota seviyesinde kalibrasyonu korurken (aggregated PICP=0.983) yorumlanabilir belirsizlik atfı sağlar. Normalized CP, segment bazında uyarlanmış interval genişlikleri üretir (4.3× heterojenlik). En yüksek belirsizlikli 5 segment, toplam rota belirsizliğinin %22'sini oluşturur. Bu, operatörlerin hangi segmentlerin en çok risk taşıdığını belirlemesini sağlar."

---

## MODÜL 8: Evaluation Metrics Derinlemesine
> *"Bu metrikleri neden seçtiniz ve ne anlama geliyorlar?"*

### 8.1 Point Prediction Metrics
- [ ] MAE: ortalama mutlak hata — yorumlanması kolay (saniye cinsinden)
- [ ] RMSE: büyük hataları daha çok cezalandırır (karesi alınıyor)
- [ ] MAPE: yüzdesel hata — ölçek bağımsız karşılaştırma

### 8.2 Uncertainty Calibration Metrics
- [ ] **PICP** (Prediction Interval Coverage Probability): gerçek değerlerin interval içine düşme oranı
  - Hedef: 1 - α (ör. 0.90). Düşükse → miscalibration, yüksekse → interval çok geniş
- [ ] **MPIW** (Mean Prediction Interval Width): ortalama interval genişliği (saniye)
  - Düşük → daha bilgilendirici, yüksek → az bilgilendirici
- [ ] **Calibration Error**: |PICP - target_coverage| — sıfıra ne kadar yakınsa o kadar iyi
- [ ] **Winkler Score**: hem coverage hem width'i birleştirir
  - Width + (2/α) × miss_penalty — düşük daha iyi
- [ ] **CWC** (Coverage Width-based Criterion): width × coverage penalty çarpanı

### 8.3 Metrikler Arası Trade-off
- [ ] PICP vs MPIW: "sonsuz genişlikte interval %100 coverage verir ama işe yaramaz"
- [ ] İdeal: yüksek PICP + düşük MPIW = dar ama kapsayıcı interval
- [ ] Winkler score bu dengeyi tek sayıda yakalar

---

## MODÜL 9: Literatür ve Araştırma Bağlamı
> *"Mevcut çalışmalardan farkınız ne?"*

### 9.1 Point Prediction Literatürü
- [ ] LSTM, GRU, GNN, Seq2Seq modelleri — yüksek doğruluk ama belirsizlik yok
- [ ] XGBoost/LightGBM — verimli, robust ama deterministik
- [ ] Bu tezin farkı: doğruluk değil, belirsizlik kalibrasyonu odaklı

### 9.2 Uncertainty Estimation Yaklaşımları
- [ ] Bayesian (Rodriguez-Deniz & Villani, 2022): MCMC → hesaplama maliyeti yüksek
- [ ] Kalman Filter (Achar et al., 2019; Schwinger et al., 2021): Gaussian varsayımı
- [ ] Bu tez: Conformal Prediction → distribution-free, finite-sample garantili

### 9.3 Non-Stationarity Çözümleri
- [ ] iETA (Han et al., 2023): incremental learning
- [ ] Transfer learning (Aemmer et al., 2024): GTFS standardizasyonu
- [ ] Bu tez: online conformal calibration → model değil, belirsizlik tahminini adapte et

### 9.4 Bu Tezin Özgün Katkıları
- [ ] Distribution-free belirsizlik tahmini + temporal drift analizi (ilk kapsamlı ampirik çalışma)
- [ ] Online vs static conformal karşılaştırması (transit domain'de)
- [ ] Segment-level decomposition ile spatial risk attribution
- [ ] Multi-layered framework: prediction + calibration + attribution

---

## MODÜL 10: Potansiyel Savunma Soruları
> *Jüri üyelerinin sorabileceği kritik sorulara hazırlık*

### 10.1 Metodoloji Soruları
- [ ] "Neden XGBoost? Derin öğrenme modeli deneseydiniz ne olurdu?" ★ DETAYLI CEVAP GEREKLİ — aşağıya bak
  > **Hazır cevap şablonu**:
  > 1. Tezin odağı en iyi point predictor bulmak değil, **uncertainty calibration** incelemek
  > 2. Conformal prediction **model-agnostic** — hangi modeli kullanırsan kullan CP aynı şekilde çalışır
  > 3. XGBoost'un avantajları: (a) tabular veriler için SOTA performans, (b) hızlı train/inference,
  >    (c) feature importance desteği, (d) literatürde ETA için yaygın kullanım (refs 12, 16)
  > 4. Deep learning (LSTM/GRU) dense sequential veri gerektirir — bu veri setinde segment bazlı
  >    tabular yapı var, sequential model avantajı sınırlı
  > 5. "Model değiştirmek CP sonuçlarını değiştirir mi?" → Nonconformity score dağılımı değişir,
  >    ama CP'nin coverage guarantee'si model bağımsızdır (exchangeability sağlandığı sürece)
  > 6. Future work olarak: GRU/LSTM ile karşılaştırma önerebilirsiniz
- [ ] "Conformal prediction'ın exchangeability varsayımı ihlal edildiğinde ne olur?"
- [ ] "Online CP'de ground truth'u nasıl elde ediyorsunuz? Gerçek zamanlı senaryoda bu mümkün mü?"
- [ ] "Segment aggregation'da independence assumption ne kadar gerçekçi?"
- [ ] "Neden Bonferroni correction çok muhafazakar sonuç veriyor?"

### 10.2 Sonuçlar Soruları
- [ ] "%90 hedefine neden hiçbir online yöntem ulaşamadı?"
- [ ] "Coverage collapse neden aniden gerçekleşti, kademeli değil?"
- [ ] "Aggregated segment coverage (%98.3) neden bu kadar yüksek? Over-conservative mi?"
- [ ] "Hourly vs daily update — neden marginal fark?"

### 10.3 Generalizasyon Soruları
- [ ] "Tek şehir, 3 rota, 55 gün — sonuçlar genellenebilir mi?"
- [ ] "Farklı bir modelle (ör. LSTM) sonuçlar değişir miydi?"
- [ ] "Mevsimsel drift (kış/yaz) bu veri setinde test edilemez — sınırlama mı?"
- [ ] "Gerçek zamanlı bir sistemde bu framework nasıl deploy edilirdi?"

### 10.4 Etik ve Pratik Sorular
- [ ] "Yanlış coverage tahmini hangi operasyonel sonuçlara yol açabilir?"
- [ ] "Yolcular bu belirsizlik bilgisini nasıl kullanabilir?"
- [ ] "Hesaplama maliyeti gerçek zamanlı kullanım için uygun mu?"

---

## ÇALIŞMA TAKVİMİ (Önerilen)

| Gün | Modül | Tahmini Süre | Öncelik |
|-----|-------|-------------|---------|
| 1 | Modül 0: Temeller | 2-3 saat | Ön koşul |
| 2 | Modül 1: Problem & Motivasyon | 1-2 saat | Yüksek |
| 3 | Modül 2: Veri & Preprocessing | 1-2 saat | Orta |
| 4 | Modül 3: Feature Engineering | 1-2 saat | Orta |
| 5-6 | Modül 4: Conformal Prediction ★★★ | 3-4 saat | KRİTİK |
| 7 | Modül 5: Experiment 1 Sonuçları | 1-2 saat | Yüksek |
| 8 | Modül 6: Experiment 2 Sonuçları | 1-2 saat | Yüksek |
| 9 | Modül 7: Experiment 3 Sonuçları | 1-2 saat | Yüksek |
| 10 | Modül 8: Evaluation Metrics | 1 saat | Orta |
| 11 | Modül 9: Literatür Bağlamı | 1-2 saat | Orta |
| 12 | Modül 10: Savunma Provası | 2-3 saat | KRİTİK |

**Toplam**: ~18-26 saat

---

## NOT
- Her modülün başında bilgi seviyeni ölçen sorular sorulacak
- Eksik temel varsa önce o tamamlanacak
- Her modülün sonunda anlama kontrolü yapılacak
- ★★★ işaretli modüller en fazla zaman ayrılması gereken kritik konular