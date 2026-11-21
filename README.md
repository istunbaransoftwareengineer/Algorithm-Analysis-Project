# CIFAR-10 Derin Öğrenme Projesi

MLP, Small CNN ve Medium CNN, CIFAR-10 üzerinde aynı veri bölmeleriyle CPU'da karşılaştırılır. Dört eğitim boyutunda, beş tohumla toplam 60 deney yapılır. Her deneme, resmî 10.000 görüntülük test kümesinde değerlendirilir.

## Dosyalar

- **benchmark.py** veri setini hazırlar, deneyleri sırayla çalıştırır ve CSV dosyalarını üretir.
- **models.py** üç PyTorch mimarisini tanımlar.
- **plot_results.py** deney özetinden makaledeki grafikleri üretir.
- **results/runs.csv** her denemenin sonuçlarını içerir.
- **results/summary.csv** model ve eğitim boyutu başına ortalama ve örneklem standart sapmasını verir.
- **results/environment.json** deney ayarları, donanım, sürümler ve veri arşivinin sağlamasını kaydeder.
- **report/** PDF makaleyi ve düzenlenebilir IEEE LaTeX kaynaklarını içerir.

## Deneyi yeniden çalıştırma

Python 3.12 ile ZIP'i açtığınız klasörden aşağıdaki komutları çalıştırın. Paketleri tercihen ayrı bir Python ortamına kurun.

    python -m pip install -r requirements.txt -r requirements-figures.txt
    python benchmark.py --base-power-w 45 --output reproduced-results
    python plot_results.py --summary reproduced-results/summary.csv --out reproduced-results/figures

İlk çalıştırmada program, [resmî CIFAR-10 Python arşivini](https://www.cs.toronto.edu/~kriz/cifar.html) indirip MD5 ve SHA-256 değerlerini doğrular. Arşiv elinizdeyse, ikinci komuta aşağıdaki seçeneği ekleyebilirsiniz.

    --archive "C:\veri\cifar-10-python.tar.gz"

Deney CPU kullanır. İlk çalıştırmada yaklaşık 163 MiB arşiv indirilir ve hazırlık için yaklaşık 176 MiB geçici disk alanı ayrılır. Bu klasördeki kayıtlı deney i7-12700H işlemci, 45 W üretici baz gücü ve dört PyTorch iş parçacığıyla yürütüldü. Başka bir işlemcide enerji tahmini için o işlemcinin yayımlanmış baz gücünü girin. Aynı ayarlar kullanıldığında bile makineye ve sistem yüküne göre süre, RAM ve enerji tahmini değişebilir.

## Ölçümleri okuma

- **Zaman (ms)** geçen gerçek süredir. Eğitim ve 10.000 görüntünün tamamı için tahmin ayrı kaydedilir.
- **Tepe RAM (MiB)** ilgili aşamada gözlenen en yüksek süreç RSS değeridir. CSV ayrıca aşama başlangıcındaki RSS değerini verir.
- **Enerji tahmini (J)** işlem sürecinin CPU zamanı, işlemcinin baz gücü ve mantıksal işlemci sayısıyla hesaplanır. Donanım sayacından alınan enerji ölçümü yerine CPU etkinliğine dayalı bir göstergedir.
- **Başarı değerleri** CSV'de 0 ile 1 arasındadır. Makalede yüzde olarak gösterilir. F1, kesinlik ve duyarlılık sınıflar üzerinden makro ortalamadır.

## Makaleyi düzenleme

Grup üyeleri, kurum ve ders bilgilerini **report/authors.tex** dosyasına girin. PDF'yi yeniden oluşturmak için TeX Live veya MiKTeX'te IEEEtran kurulu olmalıdır. Aşağıdaki komutları **report** klasöründe çalıştırın.

    pdflatex paper.tex
    pdflatex paper.tex

Makale, ZIP içindeki tamamlanmış deneyin sonuçlarını içerir. Yeni koşumları makaleye aktarmak isterseniz tablolardaki ve tartışmadaki değerleri güncelleyin ve makalenin çağırdığı figürleri yenileyin. Dersin grup bilgileri e-postaya da eklenmelidir.
