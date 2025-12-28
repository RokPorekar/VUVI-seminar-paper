# Navodila za uporabo aplikacije: Optimizacija zadovoljstva potnikov

Ta aplikacija je bila razvita v okviru seminarske naloge pri predmetu **VUVI**.  
Omogoča interaktivno napovedovanje zadovoljstva potnikov na podlagi strojnega učenja  
(**Random Forest, Gradient Boosting in KNN**) ter simulacijo optimizacije letalskih storitev.

---

## 1. Sistemske zahteve

Za zagon aplikacije potrebujete nameščeno okolje **Python**  
(priporočljivo preko distribucije **Anaconda**) in naslednje knjižnice:

- `streamlit` (za uporabniški vmesnik)
- `pandas` in `numpy` (za obdelavo podatkov)
- `scipy` (za statistične teste, porazdelitve)
- `scikit-learn` (za izvajanje modelov)
- `joblib` (za nalaganje shranjenih modelov)
- `matplotlib` in `seaborn` (za vizualizacijo grafov)

---

## 2. Namestitev okolja

Če nimate nameščenih potrebnih knjižnic, odprite terminal  
(ali **Anaconda Prompt**) in zaženite naslednji ukaz:

```bash
pip install streamlit pandas numpy scikit-learn scipy joblib matplotlib seaborn
```

## 3. Struktura datotek

Aplikacija za pravilno delovanje potrebuje naslednjo strukturo v mapi:

```
/vasa-mapa-projekta
├── app.py                      # Glavna koda Streamlit aplikacije
├── classification_models/      # Mapa s shranjenimi modeli
│   ├── model_RF.pkl            # Random Forest model
│   ├── model_GB.pkl            # Gradient Boosting model
│   └── model_KNN.pkl           # K-Nearest Neighbors model
└── (podatki.csv)               # Opcijsko: vaša testna baza za skupinsko napoved
```

## 4. Zagon aplikacije

Odprite terminal ali Anaconda Prompt.

Z ukazom cd se pomaknite v mapo, kjer se nahaja datoteka app.py.

Primer:

```bash
cd C:\Uporabniki\Ime\Seminarska_naloga
```

Zaženite aplikacijo z naslednjim ukazom:

```bash
streamlit run app.py
```

Aplikacija se bo samodejno odprla v vašem privzetem spletnem brskalniku
(običajno na naslovu http://localhost:8501).

## 5. Navodila za uporabo vmesnika

### 5.1 Posamezna napoved (Tab 1)

Izbira modela:

- V levem stranskem meniju izberite enega izmed treh naučenih modelov.

Vnos podatkov:

- S pomočjo drsnikov nastavite ocene storitev (od 1 do 5) in podatke o potniku
(starost, razdalja, razred).

Rezultat:

- Aplikacija bo takoj izračunala verjetnost zadovoljstva in prikazala profil izbranega vzorca.

Shranjevanje:

- S klikom na gumb »Prenesi trenutni scenarij v CSV« lahko shranite trenutne nastavitve
za kasnejšo analizo.

### 5.2 Skupinska napoved in optimizacija (Tab 2)

Simulacija optimizacije:

- Izberite posamezno storitev (npr. WiFi) in opazujte graf, kako se verjetnost zadovoljstva
spreminja z izboljševanjem ocene.

- Rdeča črta (50 %) predstavlja mejo, kjer potnik postane zadovoljen.

Demografski vplivi:

- Spodaj si lahko ogledate, kako starost, razdalja leta in tip potovanja vplivajo
na končni rezultat.
