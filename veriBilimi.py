import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# veri = pd.read_csv("olimpiyatlar.csv")
# # print(veri.info())
# veri.rename(columns={'ID'    : 'id', 
#                      'Name'  : 'isim', 
#                      'Sex'   : 'cinsiyet', 
#                      'Age'   : 'yas', 
#                      'Height': 'boy', 
#                      'Weight': 'kilo', 
#                      'Team'  : 'takim', 
#                      'NOC'   : 'uok', 
#                      'Games' : 'oyunlar',
#                      'Year'  : 'yil', 
#                      'Season': 'sezon', 
#                      'City'  : 'sehir',
#                      'Sport' : 'spor',
#                      'Event' : 'etkinlik',
#                      'Medal' : 'madalya'}, inplace=True)
# # print(veri.head(2))


# veri= veri.drop(["id","oyunlar"],axis=1)
# # print(veri.head())


# essiz_etkinlik = pd.unique(veri.etkinlik)
# # print("Eşsiz etkinlik sayısı: {}".format(len(essiz_etkinlik)))
# # print(essiz_etkinlik[:10])


# # # her bir etkinliği iteratif olarak dolaş
# # # etkinlik özelinde boy ve kilo ortalamalarını bul
# # # etkinlik özelinde boy ve kilo da kayıp olan değerlere ortalama boy ve kilo değerlerini eşitle

# # veri_gecici = veri.copy() # gerçek veriyi kaybetmemek için veri_gecici değişkeni belirle
# # boy_kilo_liste = ["boy", "kilo"]
# # for e in essiz_etkinlik: # etkinlik listesi içerisinde dolaş

# #     # etkinlik filtresi oluştur
# #     etkinlik_filtre = veri_gecici.etkinlik == e
# #     # veriyi etkinliğe göre filtrele
# #     veri_filtreli = veri_gecici[etkinlik_filtre]

# #     # boy ve kilo için etkinlik özelinde ortalama bul
# #     for s in boy_kilo_liste:
# #         ortalama = np.round(np.mean(veri_filtreli[s]),2)
# #         if ~np.isnan(ortalama): # eğer etkinlik özelinde ortalama varsa
# #             veri_filtreli[s] = veri_filtreli[s].fillna(ortalama)
# #         else: # etkinlik özelinde ortalama yoksa tüm veri için ortalama bul
# #             tum_veri_ortalamasi = np.round(np.mean(veri[s]),2)
# #             veri_filtreli[s] = veri_filtreli[s].fillna(tum_veri_ortalamasi)
# #     # etkinlik özelinde kayıp değerleri doldurulmuş veriyi veri geçiciye eşitle            
# #     veri_gecici[etkinlik_filtre] = veri_filtreli

# # # kayıp değerleri giderilmiş geçici veriyi gerçek veri değişkenine eşitle
# veri = veri_gecici.copy() 
# # print(veri.info()) # boy ve kilo sütunlarında kayıp değer sayısına bak

# # 2. Yaş sütununda bulunan kayıp veriyi veri setinin yaş ortalamasına göre dolduracağız.
# # yas değişkeni tanımlı olmayan örnekleri bul, 
# # tilda işareti ile tersini al
# # yaş değişkeni tanımlı olan örnekleri bulmak için filtre oluştur
# # yas_ortalamasi = np.round(np.mean(veri.yas),2)
# # print(f"Yaş ortlaması: {yas_ortalamasi}")
# veri["yas"] = veri["yas"].fillna(yas_ortalamasi)
# # print(veri.info())


# # 3. Madalya alamayan sporcuları veri setinden çıkaracağız.
# # # toplamda 231333 tane örnek için madalya değişkeni tanımlı değil
# madalya_degiskeni = veri["madalya"]
# # print(pd.isnull(madalya_degiskeni).sum)
# # madalya değişkeni tanımlı olmayan örnekleri bul, (NaN) 
# # tilda işareti ile tersini al
# # madalya değişkeni tanımlı olan örnekleri bulmak için filtre oluştur
# madalya_degiskeni_filtresi = ~pd.isnull(madalya_degiskeni)
# veri = veri[madalya_degiskeni_filtresi]
# # print(veri.head(5))
# veri.to_csv("olimpiyatlar_temizlenmis.csv", index=False)

veri = pd.read_csv("olimpiyatlar_temizlenmis.csv")
# öncelikli olarak histogram grafiğini elde edeceğimiz metodumuzu yazalım.
def plotHistogram(degisken):
    """
        Girdi: Değişken/sütun ismi
        Çıktı: Histogram grafiği
    """
    
    plt.figure()
    plt.hist(veri[degisken], bins = 85, color = "orange")
    plt.xlabel(degisken)
    plt.ylabel("Frekans")
    plt.title(f"Veri Sıklığı - {degisken}")
    plt.show()
def plotBar(degisken, n=5):
    veri_ = veri[degisken]
    veri_sayma = veri_.value_counts()
    veri_sayma = veri_sayma[:n]
    plt.figure()
    plt.bar(veri_sayma.index, veri_sayma, color="orange")
    plt.xticks(veri_sayma.index, veri_sayma.index.values)
    plt.ylabel("Frekans")
    plt.title("Veri Sıklığı")
    plt.show()
    print(f"{degisken}: \n {veri_sayma}")
# sayısal değişkenler için histogram çizdirelim
# kategorik_degisken = ["isim", "cinsiyet", "takim", "uok", "sezon", "sehir", "spor", "etkinlik", "madalya"]
# for i in kategorik_degisken:
#     plotBar(i)
# sayısal değişkenler için histogram çizdirelim
# sayisal_degisken = ["yas", "boy", "kilo", "yil"]
# for i in sayisal_degisken:
#     plotHistogram(i)
# yas değişkeni için filtreyi uygulayıp sonra kutu grafiği çizdirelim
# aslında kutu grafiği çizdirmenin daha etkili yolları var, bu yolları görselleştirme bölümünde öğreneceğiz
# plt.boxplot(veri.yas)
# plt.title(" Yaş Değişkeni için Kutu Grafiği")
# plt.xlabel("yas")
# plt.ylabel("Değer")
# plt.show()

erkek = veri[veri.cinsiyet == "M"]
# print(erkek.head(1))
kadin = veri[veri.cinsiyet == "F"]
# print(kadin.head(1))
# plt.figure()
# plt.scatter(kadin.boy, kadin.kilo, alpha = 0.4, label = "Kadin")
# plt.scatter(erkek.boy, erkek.kilo, alpha = 0.4, label = "Erkek")
# plt.xlabel("Boy")
# plt.ylabel("Kilo")
# plt.title("Boy ve Kilo Arasındaki İlişki")
# plt.legend()
# plt.show()
# sayisal veriler arasında ilişki incelemesi
# print(veri.loc[:,["yas","boy","kilo"]].corr()) # korelasyon tablosu
veri_gecici = veri.copy()
veri_gecici = pd.get_dummies(veri_gecici, columns=["madalya"])
# print(veri_gecici.head(2))
# print(veri_gecici.loc[:,["yas","madalya_Bronze", "madalya_Gold","madalya_Silver"]].corr())
# print(veri_gecici[["takim","madalya_Gold", "madalya_Silver", "madalya_Bronze"]].groupby(["takim"], as_index = False).sum().sort_values(by="madalya_Gold",ascending = False)[40:41])
# print(veri_gecici[["sehir","madalya_Gold", "madalya_Silver", "madalya_Bronze"]].groupby(["sehir"], as_index = False).sum().sort_values(by="madalya_Gold",ascending = False)[:])
# print(veri_gecici[["cinsiyet","madalya_Gold", "madalya_Silver", "madalya_Bronze"]].groupby(["cinsiyet"], as_index = False).sum().sort_values(by="madalya_Gold",ascending = False)[:10])
veri_pivot = veri.pivot_table(index="madalya", columns = "cinsiyet",
                 values=["boy","kilo","yas"], 
                aggfunc={"boy":np.mean,"kilo":np.mean,"yas":[min, max, np.std]})
# print(veri_pivot.head())

def anomaliTespiti(df,ozellik):
    outlier_indices = []

    for c in ozellik:
        # 1. çeyrek
        Q1 = np.percentile(df[c],25)
        # 3. çeyrek
        Q3 = np.percentile(df[c],75)
        # IQR: Çeyrekler açıklığı
        IQR = Q3 - Q1
        # aykırı tespiti için çarpan
        outlier_step = IQR * 1.5
        # aykırıyı ve aykırı indeksini tespit et
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # indeksleri depola
        outlier_indices.extend(outlier_list_col)

    # eşsiz aykırı değerleri bul
    outlier_indices = Counter(outlier_indices)
    return [i for i, v in outlier_indices.items() if v > 1]

veri_anamoli = veri.loc[anomaliTespiti(veri,["yas","kilo","boy"])]
# print(veri_anamoli.spor.value_counts())

# plt.figure()
# plt.bar(veri_anamoli.spor.value_counts().index,veri_anamoli.spor.value_counts().values)
# plt.xticks(rotation = 30)
# plt.title("Anomali Görünen Spor Branşları")
# plt.ylabel("Frekans")
# plt.grid(True,alpha = 0.5)
# plt.show()
veri_gym = veri_anamoli[veri_anamoli.spor == "Gymnastics"]
# print(veri_gym)
# print(veri_gym.etkinlik.value_counts())
veri_zaman = veri.copy()
# print(veri_zaman.head(3))
essiz_yillar = veri_zaman.yil.unique()
# print(essiz_yillar)
dizili_array = np.sort(veri_zaman.yil.unique())
# print(dizili_array)
# plt.figure()
# plt.scatter(range(len(dizili_array)),dizili_array)
# plt.grid(True)
# plt.ylabel("Yıllar")
# plt.title("Olimpiyatlar Çift Yıllarda Düzenlenir")
# plt.show()
tarih_saat_nesnesi = pd.to_datetime(veri_zaman["yil"], format="%Y")
# print(type(tarih_saat_nesnesi))
# print(tarih_saat_nesnesi.head())
veri_zaman["tarih_saat"] = tarih_saat_nesnesi
# print(veri_zaman.head(3))
# tarih_saat sütununda bulunan datetime veri tipine ait veriyi, asıl verinin indeksi yapalım
# pandas kütüphanesinde indeksi datetime veri tipi olan veri setleri ile çalışmak için özel yapılar bulunmaktadır.
# bu nedenle amacımız olan indeksi datetime veri tipi yapma çalışmamzız gerçekleşmiş oluyor.
veri_zaman = veri_zaman.set_index("tarih_saat")
veri_zaman.drop(["yil"],axis = 1,inplace= True)
# print(veri_zaman.head(3))
periyodik_veri = veri_zaman.resample("2A").mean() # 2 yıllık periyotlar halinde ortalama değerleri al
periyodik_veri.dropna(axis=0, inplace=True) 
# print(periyodik_veri.head())
plt.figure()
periyodik_veri.plot()
plt.title("Yıllara göre ortalama yaş, boy, kilo değişimi")
plt.xlabel("yıl")
plt.grid(True)
plt.show()