import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. KONFIGURACIJA 
st.set_page_config(page_title="Flight Satisfaction Optimizer", layout="wide")
st.title("Optimizacija zadovoljstva potnikov")

# Točen seznam iz tvojega modela
top_15_features = [
    'Online boarding', 'Inflight wifi service', 'Inflight entertainment', 
    'Seat comfort', 'Flight Distance', 'Leg room service', 
    'Ease of Online booking', 'Age', 'On-board service', 'Baggage handling', 
    'Cleanliness', 'Checkin service', 'Customer Type_disloyal Customer', 
    'Type of Travel_Personal Travel', 'Class_Eco'
]

# 2. NALAGANJE MODELOV 
@st.cache_resource
def load_models():
    return {
        "Random Forest (RF)": joblib.load('classification_models/model_RF.pkl'),
        "Gradient Boosting (GB)": joblib.load('classification_models/model_GB.pkl'),
        "K-Nearest Neighbors (KNN)": joblib.load('classification_models/model_KNN.pkl')
    }

models_dict = load_models()

# 3. STRANSKA VRSTICA (Side Bar) 
st.sidebar.header("Izbira modela")
selected_model_name = st.sidebar.selectbox("Model:", list(models_dict.keys()))
model = models_dict[selected_model_name]

st.sidebar.divider()
st.sidebar.header("Vnos parametrov")

input_data = {}

# Numerične ocene (0-5)
st.sidebar.subheader("Ocene storitev")
service_cols = ['Online boarding', 'Inflight wifi service', 'Inflight entertainment', 
                'Seat comfort', 'Leg room service', 'Ease of Online booking', 
                'On-board service', 'Baggage handling', 'Cleanliness', 'Checkin service']

for col in service_cols:
    input_data[col] = st.sidebar.slider(col, 1, 5, 3)

# Številčni vnosi
st.sidebar.subheader("Podatki o potniku")
input_data['Flight Distance'] = st.sidebar.number_input("Flight Distance", value=1000)
input_data['Age'] = st.sidebar.slider("Starost", 7, 85, 35)

# Binarne spremenljivke (Bolj intuitivna izbira)
st.sidebar.subheader("Kategorije potovanja")

# 1. Tip stranke
cust_type = st.sidebar.radio("Tip stranke:", ["Zvesta stranka (Loyal)", "Nezvesta stranka (Disloyal)"])
input_data['Customer Type_disloyal Customer'] = 1 if cust_type == "Nezvesta stranka (Disloyal)" else 0

# 2. Namen potovanja
travel_purpose = st.sidebar.radio("Namen potovanja:", ["Poslovno potovanje", "Osebno potovanje"])
input_data['Type of Travel_Personal Travel'] = 1 if travel_purpose == "Osebno potovanje" else 0

# 3. Potovalni razred
travel_class = st.sidebar.selectbox("Potovalni razred:", ["Business / Eco Plus", "Eco"])
input_data['Class_Eco'] = 1 if travel_class == "Eco" else 0

# Pretvori v DataFrame v pravilnem vrstnem redu
input_df = pd.DataFrame([input_data])[top_15_features]

# 4. OSREDNJI DEL: NAPOVED 
tab1, tab2 = st.tabs(["Posamezna Napoved", "Skupinska Napoved & Optimizacija"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trenutni status")
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
        
        if pred == 1:
            st.success("Potnik bo verjetno ZADOVOLJEN")
        else:
            st.error("Potnik bo verjetno NEZADOVOLJEN")
        
        st.metric("Verjetnost zadovoljstva", f"{prob:.1%}")

        st.write("---")
        st.write("### Shranjevanje scenarija")
        
        # Pripravimo podatke za izvoz (vsi parametri + napoved)
        export_df = input_df.copy()
        export_df['Prediction'] = "Satisfied" if pred == 1 else "Neutral/Dissatisfied"
        export_df['Probability'] = f"{prob:.4f}"
        export_df['Selected_Model'] = selected_model_name

        csv_single = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Prenesi trenutni scenarij v CSV",
            data=csv_single,
            file_name="izbrani_vzorec_optimizacija.csv",
            mime="text/csv"
        )

    with col2:
        # GRAF 1: PROFIL STORITEV (Vodoravni stolpci) 
        fig, ax = plt.subplots(figsize=(8, 6))
        current_scores = {k: v for k, v in input_data.items() if k in service_cols}
        
        bars = ax.barh(list(current_scores.keys()), list(current_scores.values()), color='skyblue')
        
        # Nastavitev osi od 1 do 5
        ax.set_xlim(1, 5) 
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_title("Trenutni profil ocen potnika")
        
        # Dodajanje vrednosti na konce stolpcev
        for bar in bars:
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{int(bar.get_width())}', va='center', fontweight='bold')
            
        st.pyplot(fig)

with tab2:
    st.subheader("Simulacija optimizacije")
    
    # 1. DEL: SIMULACIJA STORITEV (Tvoja obstoječa koda z izboljšavami)
    target_sim = st.selectbox("Analiziraj vpliv spremembe za:", service_cols)
    
    sim_range = [1, 2, 3, 4, 5]
    sim_values = []
    for val in sim_range:
        temp_df = input_df.copy()
        temp_df[target_sim] = val
        sim_values.append(model.predict_proba(temp_df)[0][1])
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(sim_range, sim_values, marker='o', color='green', linewidth=3, markersize=8)
    ax2.fill_between(sim_range, sim_values, color='green', alpha=0.1)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, label="Meja zadovoljstva (50%)")
    
    ax2.set_xlim(1, 5)
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel(f"Nastavljena ocena za: {target_sim}")
    ax2.set_ylabel("Verjetnost zadovoljstva")
    ax2.set_title(f"Vpliv storitve '{target_sim}' na verjetnost zadovoljstva")
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    for i, v in enumerate(sim_values):
        ax2.text(sim_range[i], v + 0.03, f"{v:.1%}", ha='center', fontweight='bold')
    
    st.pyplot(fig2)
    st.info(f"💡 **Nasvet:** Graf prikazuje točko prevoja, kjer investicija v '{target_sim}' dejansko spremeni potnika iz nezadovoljnega v zadovoljnega.")

    # 2. DEL: SIMULACIJA PODATKOV O POTNIKU (Novo!) 
    st.divider()
    st.subheader("Simulacija demografskih in potovalnih faktorjev")
    st.write("Preverite, kako na zadovoljstvo vplivajo faktorji, na katere podjetje nima neposrednega vpliva, a so ključni za segmentacijo.")

    col_sim1, col_sim2 = st.columns(2)

    with col_sim1:
        # Simulacija STAROSTI
        age_range = np.linspace(7, 85, 20)
        age_probs = []
        for a in age_range:
            temp_df = input_df.copy()
            temp_df['Age'] = a
            age_probs.append(model.predict_proba(temp_df)[0][1])
        
        fig_age, ax_age = plt.subplots()
        ax_age.plot(age_range, age_probs, color='orange', linewidth=2)
        ax_age.set_title("Vpliv STAROSTI na zadovoljstvo")
        ax_age.set_xlabel("Leta")
        ax_age.set_ylabel("Verjetnost")
        ax_age.grid(True, alpha=0.2)
        st.pyplot(fig_age)

    with col_sim2:
        # Simulacija RAZDALJE LETA
        dist_range = np.linspace(100, 4000, 20)
        dist_probs = []
        for d in dist_range:
            temp_df = input_df.copy()
            temp_df['Flight Distance'] = d
            dist_probs.append(model.predict_proba(temp_df)[0][1])
        
        fig_dist, ax_dist = plt.subplots()
        ax_dist.plot(dist_range, dist_probs, color='purple', linewidth=2)
        ax_dist.set_title("Vpliv RAZDALJE LETA na zadovoljstvo")
        ax_dist.set_xlabel("Milje")
        ax_dist.grid(True, alpha=0.2)
        st.pyplot(fig_dist)

    # 3. DEL: ANALIZA BINARNIH KATEGORIJ 
    st.write("### Vpliv kategorije potnika")
    
    # Primerjava: Kaj če bi bil ta isti potnik v drugem razredu ali na drugem tipu leta?
    cat_features = ['Customer Type_disloyal Customer', 'Type of Travel_Personal Travel', 'Class_Eco']
    cat_diffs = []

    for feat in cat_features:
        # Verjetnost če je vrednost 0
        df0 = input_df.copy()
        df0[feat] = 0
        p0 = model.predict_proba(df0)[0][1]
        
        # Verjetnost če je vrednost 1
        df1 = input_df.copy()
        df1[feat] = 1
        p1 = model.predict_proba(df1)[0][1]
        
        cat_diffs.append({'Faktor': feat, 'Verjetnost (0)': p0, 'Verjetnost (1)': p1})

    df_cat = pd.DataFrame(cat_diffs)
    
    # Grafični prikaz razlike
    fig_cat, ax_cat = plt.subplots(figsize=(10, 4))
    x = np.arange(len(cat_features))
    width = 0.35
    
    ax_cat.bar(x - width/2, df_cat['Verjetnost (0)'], width, label='Vrednost 0 (npr. Loyal / Business)', color='silver')
    ax_cat.bar(x + width/2, df_cat['Verjetnost (1)'], width, label='Vrednost 1 (npr. Disloyal / Personal / Eco)', color='salmon')
    
    ax_cat.set_xticks(x)
    ax_cat.set_xticklabels(['Lojalnost', 'Tip potovanja', 'Razred'])
    ax_cat.set_ylabel('Verjetnost zadovoljstva')
    ax_cat.set_title('Primerjava vpliva binarnih kategorij na trenutni vzorec')
    ax_cat.legend()
    st.pyplot(fig_cat)