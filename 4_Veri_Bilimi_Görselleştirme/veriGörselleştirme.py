import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
veri = pd.read_csv('olimpiyatlar_temizlenmis.csv')
def sacilimGrafik():
    sns.scatterplot(x="boy", y="kilo", data=veri)
    plt.title("Boy ve Kilo Dağılımı - Beyaz Izgara Tema")
    plt.show()
    
# sacilimGrafik()

sns.set_style("white")
# sacilimGrafik()

#KATEGORİK

# sns.scatterplot(x="boy", y="kilo", hue="madalya",data=veri)
# plt.title("Madalyaya Göre Boy ve Kilo Dağılımı")
# plt.show()

#DOĞRUSAL REGRESYON

# sns.regplot(x="boy", y="kilo",data=veri, marker="+", scatter_kws={"alpha":0.2})
# plt.title("Boy ve Kilo Dağlımı")
# plt.show()
# sns.scatterplot(x="boy",y="kilo",hue="madalya",data=veri, palette="Set1")
# plt.title("Madalyaya Göre Boy ve Kilo Dağılımı")
# plt.show()
# sns.scatterplot(x="boy",y="kilo",hue="madalya",data=veri, palette="rocket")
# plt.title("Madalyaya Göre Boy ve Kilo Dağılımı")
# plt.show()

#ÇİZGİ GRAFİĞİ

# sns.lineplot(x="boy",y="kilo",data=veri)
# plt.title("Boy ve Kilo")
# plt.show()

#KATEGORİK

# sns.lineplot(x="boy",y="kilo",hue="cinsiyet",data=veri)
# plt.title("Cinsiyete göre Boy ve Kilo")
# plt.show()

#HİSTOGRAM KATEGORİK

# sns.displot(veri, x="kilo", hue="cinsiyet")
# plt.ylabel("Frekans")
# plt.title("Cinsiyete Göre Kilo Histogram")
# # plt.show()
# sns.displot(veri,x="kilo", col="cinsiyet",multiple="dodge")
# plt.show()

#İKİ BOYUTLU HİSTOGRAM

# sns.displot(veri, x="kilo", y="boy", kind="kde")
# plt.xlabel("kilo")
# plt.ylabel("boy")
# plt.title("Kilo - Boy Histogram")
# plt.show()
# sns.displot(veri, x="kilo", y="boy", hue="cinsiyet",kind="kde")
# plt.xlabel("kilo")
# plt.ylabel("boy")
# plt.title("Cinsiyete göre Kilo - Boy Histogram")
# plt.show()
# sns.kdeplot(data=veri, x="kilo", y="boy", hue="cinsiyet", fill=True)
# plt.xlabel("kilo")
# plt.ylabel("boy")
# plt.title("Cinsiyete göre Kilo - Boy Histogram")
# plt.show()

#ÇUBUK GRAFİĞİ

# sns.barplot(x="madalya", y="boy", data=veri)
# plt.title("Madalyaya Göre Boy Grafikleri")
# plt.show()
# sns.barplot(x="madalya", y="boy", data=veri, hue="cinsiyet")
# plt.title("Madalyaya Göre Boy Grafikleri")
# plt.show()
# sns.catplot(x="madalya", y="yas", hue="cinsiyet",col="sezon",
#             data=veri, kind="bar", height=4, aspect=.7)
# plt.show()

#KUTU GRAFİĞİ

# sns.boxplot(x="sezon", y="boy", data=veri)
# plt.show()
# sns.boxplot(x="sezon", y="boy", hue="cinsiyet", data=veri, palette="Set2")
# plt.show()
# veri_gecici = veri.loc[:,["yas","boy","kilo"]]
# sns.boxplot(data=veri_gecici, orient="h", palette="Set2")
# plt.show()
# sns.catplot(x="sezon", y="boy",hue="cinsiyet", col="madalya", data=veri, kind="box", height=4, aspect=.7)
# plt.show()

#SICAKLIK HARİTASI

sns.heatmap