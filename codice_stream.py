import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.title('PROGETTO BUSINESS INTELLIGENCE')
st.subheader('ANALISI DELLE CATTURE')

# Caricamento dei dati
dati_cattura_cicalino = pd.read_csv('dati_cattura_cicalino.csv', encoding='ISO-8859-1', sep=',')
dati_cattura_imola = pd.read_csv('dati_cattura_imola.csv', encoding='ISO-8859-1', sep=',')
meteo_cicalino = pd.read_csv('meteo_cicalino.csv', encoding='ISO-8859-1', sep=';')
meteo_imola = pd.read_csv('dmeteo_imola.csv', encoding='ISO-8859-1', sep=';')


# Normalizzazione dei nomi delle colonne
if meteo_cicalino is not None:
    meteo_cicalino.columns = meteo_cicalino.columns.str.strip().str.lower()
if meteo_imola is not None:
    meteo_imola.columns = meteo_imola.columns.str.strip().str.lower()
if dati_cattura_cicalino is not None:
    dati_cattura_cicalino.columns = dati_cattura_cicalino.columns.str.strip().str.lower()
if dati_cattura_imola is not None:
    dati_cattura_imola.columns = dati_cattura_imola.columns.str.strip().str.lower()

# Conversione delle colonne 'datetime' in formato datetime per ogni DataFrame
meteo_cicalino['datetime'] = pd.to_datetime(meteo_cicalino['datetime'], format='%d/%m/%Y %H:%M', dayfirst=True)
meteo_imola['datetime'] = pd.to_datetime(meteo_imola['datetime'], format='%d/%m/%Y %H:%M', dayfirst=True)
dati_cattura_cicalino['datetime'] = pd.to_datetime(dati_cattura_cicalino['datetime'])
dati_cattura_imola['datetime'] = pd.to_datetime(dati_cattura_imola['datetime'])

# Impostare 'datetime' come indice per tutti i DataFrame
dataframes = [meteo_cicalino, meteo_imola, dati_cattura_cicalino, dati_cattura_imola]
for df in dataframes:
    df.set_index('datetime', inplace=True)

# Rimuovere la colonna 'datetime' in più se presente (per evitare ambiguità)
for df in dataframes:
    if 'datetime' in df.columns:
        df.drop(columns='datetime', inplace=True)

# Convertire valori numerici con la virgola (',' come separatore decimale) in punti ('.')
meteo_cicalino = meteo_cicalino.replace(',', '.', regex=True)
meteo_imola = meteo_imola.replace(',', '.', regex=True)

# Conversione delle colonne in valori numerici, con gestione dei valori non validi (NaN)
meteo_cicalino = meteo_cicalino.apply(pd.to_numeric, errors='coerce')
meteo_imola = meteo_imola.apply(pd.to_numeric, errors='coerce')


#MERGE DATI
# Elaborazione dei dati di Cicalino
dati_cattura_cicalino.index = dati_cattura_cicalino.index.round('H')

# Raggruppamento per orario e somma dei valori di "Numero di insetti"
dati_cattura_grouped = dati_cattura_cicalino.groupby(dati_cattura_cicalino.index)['numero di insetti'].sum()

# Creazione del dataset per Cicalino
dati_cicalino = meteo_cicalino.copy()
dati_cicalino['numero di insetti'] = dati_cattura_grouped

# Somma delle "nuove catture per evento"
dati_nuove_catture = dati_cattura_cicalino.groupby(dati_cattura_cicalino.index)['nuove catture (per evento)'].sum()
dati_cicalino['nuove catture (per evento)'] = dati_nuove_catture

# Verifica di valori mancanti per Cicalino
#st.write("Valori mancanti in dati_cicalino dopo il merge tra dati cattura e dati meteorologici:")
#st.write(dati_cicalino.isnull().sum())

# Elaborazione dei dati di Imola
dati_cattura_imola.index = dati_cattura_imola.index.round('H')

# Raggruppamento per orario e somma dei valori di "Numero di insetti"
dati_cattura_grouped1 = dati_cattura_imola.groupby(dati_cattura_imola.index)['numero di insetti'].sum()

# Creazione del dataset per Imola
dati_imola = meteo_imola.copy()
dati_imola['numero di insetti'] = dati_cattura_grouped1

# Somma delle "nuove catture per evento"
dati_nuove_catture_imola = dati_cattura_imola.groupby(dati_cattura_imola.index)['nuove catture (per evento)'].sum()
dati_imola['nuove catture (per evento)'] = dati_nuove_catture_imola


#INTERPOLAZIONE CICALINO(new_dataset_interpolated)
# Interpolazione temporale per imputare i dati mancanti
new_dataset_interpolated = dati_cicalino.copy()
new_dataset_interpolated['numero di insetti'] = new_dataset_interpolated['numero di insetti'].interpolate(
    method='time', limit_direction='both'
)

new_dataset_interpolated['nuove catture (per evento)'] = new_dataset_interpolated['nuove catture (per evento)'].interpolate(
    method='time', limit_direction='both'
)
# Arrotondamento al numero intero più vicino per mantenere la natura discreta dei dati
new_dataset_interpolated['numero di insetti'] = new_dataset_interpolated['numero di insetti'].round().astype(int)
new_dataset_interpolated['nuove catture (per evento)'] = new_dataset_interpolated['nuove catture (per evento)'].round().astype(int)
new_dataset_interpolated.isnull().sum()



#INTERPOLAZIONE IMOLA(dati_interpolati_imola)
dati_interpolati_imola=dati_imola.copy()
dati_interpolati_imola['numero di insetti']=dati_interpolati_imola['numero di insetti'].interpolate(
    method='time',limit_direction='both'
)

dati_interpolati_imola['nuove catture (per evento)']=dati_interpolati_imola['nuove catture (per evento)'].interpolate(
    method='time',limit_direction='both'
)
dati_interpolati_imola['numero di insetti'] = dati_interpolati_imola['numero di insetti'].round().astype(int)
dati_interpolati_imola['nuove catture (per evento)'] = dati_interpolati_imola['nuove catture (per evento)'].round().astype(int)
dati_interpolati_imola.isnull().sum()

#dati cattura ha un solo altro valore mancante in low temoerature che possiamo facilmente riempire usando il valore precedente
dati_interpolati_imola['low temperature']=dati_interpolati_imola['low temperature'].fillna(method='bfill')
dati_interpolati_imola.isnull().sum()

#DEFINIAMO I DATASET CON LE VARIABILI LAGGED
#CICALINO
max_lag = 5  # Numero di lag significativi
for lag in range(1, max_lag + 1):
    new_dataset_interpolated[f'numero di insetti_lag_{lag}'] = new_dataset_interpolated['numero di insetti'].shift(lag)

# Rimuovi righe con valori mancanti dovute ai lag
new_dataset_lagged = new_dataset_interpolated.dropna()
new_dataset_lagged.head()

#IMOLA
max_lag = 5  # Numero di lag significativi
for lag in range(1, max_lag + 1):
    dati_interpolati_imola[f'numero di insetti_lag_{lag}'] = dati_interpolati_imola['numero di insetti'].shift(lag)

# Rimuovi righe con valori mancanti dovute ai lag
dati_lagged_imola = dati_interpolati_imola.dropna()
dati_lagged_imola.head()


#LAGGED NUOVE CATTURE
max_lag = 5  # Numero di lag significativi
for lag in range(1, max_lag + 1):
    new_dataset_interpolated[f'nuove catture (per evento)_lag_{lag}'] = new_dataset_interpolated['nuove catture (per evento)'].shift(lag)

# Rimuovi righe con valori mancanti dovute ai lag
new_dataset_lagged = new_dataset_interpolated.dropna()
new_dataset_lagged.head()

#IMOLA
max_lag = 5  # Numero di lag significativi
for lag in range(1, max_lag + 1):
    dati_interpolati_imola[f'nuove catture (per evento)_lag_{lag}'] = dati_interpolati_imola['nuove catture (per evento)'].shift(lag)

# Rimuovi righe con valori mancanti dovute ai lag
dati_lagged_imola = dati_interpolati_imola.dropna()
dati_lagged_imola.head()


# Creazione della barra laterale
st.sidebar.title('Seleziona il dataset')

# Opzioni per la selezione del dataset
opzioni_dataset = ['Cicalino', 'Imola']
scelta_dataset = st.sidebar.radio('Scegli il dataset:', opzioni_dataset)

# Dizionari per i dataset lagged (specifici per la sezione Modelli)
datasets_lagged = {
    "Cicalino": new_dataset_lagged,
    "Imola": dati_lagged_imola
}

# Dizionari per altri dataset (opzionali per altre sezioni)
datasets_originali = {
    "Cicalino": dati_cicalino,
    "Imola": dati_imola
}

datasets_interpolati = {
    "Cicalino": new_dataset_interpolated,
    "Imola": dati_interpolati_imola
}

if scelta_dataset:
    st.sidebar.title('Seleziona una funzione')
    funzioni = ['I nostri dati', 'EDA', 'Modelli']
    scelta_funzione = st.sidebar.selectbox('Scegli una funzione:', funzioni)

    #"I nostri dati"
    if scelta_funzione == 'I nostri dati':
        st.header('📉 I nostri dati')

        # Selettore per scegliere tra dati originali e interpolati
        tipo_dati = st.radio("Seleziona il tipo di dati:", ['Originali', 'Interpolati'])

        # Mostra il dataset scelto
        if tipo_dati == 'Originali':
            dataset_corrente = datasets_originali.get(scelta_dataset, None)
            st.subheader(" 🔢 Dati Originali")
            #st.dataframe(dataset_corrente)

            st.subheader("📊 Riepilogo Dati")
            st.write(dataset_corrente.describe().T[['mean', 'min', 'max']].rename(columns={'mean': 'Media', 'min': 'Minimo', 'max': 'Massimo'}))
            #valori mancanti
            st.write('📌 Valori mancanti per colonna:')
            st.write(dataset_corrente.isnull().sum())

        else:
            dataset_corrente = datasets_interpolati.get(scelta_dataset, None)
            st.subheader(" 🔄 Dati Interpolati")

                        # Aggiunta della spiegazione sull'interpolazione
            with st.expander("ℹ️ Cos'è l'interpolazione?"):
                st.write("""
                Abbiamo usato l’interpolazione temporale per imputare i valori mancanti nei nostri dataset.  
                Questo metodo sfrutta l'ordine cronologico dei dati per stimare i valori mancanti in modo coerente 
                con l'andamento temporale delle osservazioni.  
                Consente di interpolare i valori mancanti sia verso l'alto che verso il basso nella serie temporale.
                """)

                        # Valori mancanti dopo l'interpolazione
            st.subheader("✅ Valori mancanti dopo l'interpolazione")
            #valori_mancanti = dataset_corrente.isnull().sum()
            colonne_da_escludere = ['numero di insetti_lag_1','numero di insetti_lag_2','numero di insetti_lag_3','numero di insetti_lag_4','numero di insetti_lag_5','nuove catture (per evento)_lag_1','nuove catture (per evento)_lag_2','nuove catture (per evento)_lag_3','nuove catture (per evento)_lag_4','nuove catture (per evento)_lag_5']  # Sostituisci con i nomi delle colonne da escludere
            valori_mancanti = dataset_corrente.drop(columns=colonne_da_escludere).isnull().sum()
            st.write(valori_mancanti)

            # Se non ci sono più valori mancanti, mostra un messaggio chiaro
            if valori_mancanti.sum() == 0:
                st.success("🎉 Dopo l'interpolazione, non ci sono più valori mancanti nel dataset!")

        if dataset_corrente is not None:
            st.write(f" 🧮 Dataset selezionato: {scelta_dataset}")
            st.dataframe(dataset_corrente)
        else:
            st.error("Il dataset selezionato non è disponibile.")

    # Funzione EDA (Analisi esplorativa dei dati)
    elif scelta_funzione == 'EDA':
        st.header(' 📊 Exploratory Data Analysis (EDA)')
        dataset_corrente = datasets_interpolati.get(scelta_dataset, None)
        if dataset_corrente is not None:
             # Filtro data: selezione intervallo temporale
            st.sidebar.subheader("📅 Filtra per intervallo di tempo")
            min_date = dataset_corrente.index.min()
            max_date = dataset_corrente.index.max()

            data_inizio, data_fine = st.sidebar.date_input(
                "Seleziona l'intervallo di tempo",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )

            # Filtriamo il dataset in base alle date selezionate
            mask = (dataset_corrente.index >= pd.to_datetime(data_inizio)) & (dataset_corrente.index <= pd.to_datetime(data_fine))
            dataset_filtrato = dataset_corrente.loc[mask]
            

            # Grafico a linee
            st.subheader(' 📈 Andamento nel tempo del Numero di insetti')
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(dataset_filtrato.index, dataset_filtrato['numero di insetti'], color='blue', linewidth=2)
            ax.set_xlabel('Tempo')
            ax.set_ylabel('Insetti')
            ax.set_title(f"Line chart numero di insetti ({scelta_dataset})")
            st.pyplot(fig)

            # Distribuzione del numero di insetti
            st.subheader(' 📊 Distribuzione del Numero di Insetti')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(dataset_filtrato['numero di insetti'], bins=30, kde=True, color='blue', ax=ax)
            ax.set_xlabel('Numero di Insetti')
            ax.set_ylabel('Frequenza')
            ax.set_title('Istogramma della Distribuzione del Numero di Insetti')
            st.pyplot(fig)


            # Andamento delle altre variabili
            st.subheader("📈 Andamento delle Altre Variabili")
            variabili_disponibili = [col for col in dataset_corrente.columns if col != 'numero di insetti']
            variabile_scelta = st.selectbox(
            "Seleziona la variabile da analizzare",
            variabili_disponibili,
            index=variabili_disponibili.index('nuove catture (per evento)')  # Impostiamo "media temperatura" come valore di default
            )

             # Visualizza il grafico a linee per la variabile selezionata
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(dataset_filtrato.index, dataset_filtrato[variabile_scelta], color='green', linewidth=2)
            ax.set_xlabel('Tempo')
            ax.set_ylabel(variabile_scelta)
            ax.set_title(f"Andamento della variabile: {variabile_scelta}")
            st.pyplot(fig)


            # Matrice di correlazione
            st.subheader(' 🔗 Matrice delle correlazioni')
            # Aggiungi un checkbox per mostrare le annotazioni
            mostra_annotazioni = st.checkbox('Mostra annotazioni nella matrice di correlazione', value=True)

            # Calcolo la matrice di correlazione
            correlation_matrix = dataset_filtrato.corr()

            # Mostra la matrice di correlazione con o senza annotazioni in base alla scelta dell'utente
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=mostra_annotazioni, cmap='coolwarm', ax=ax, fmt='.2f', cbar=True)

            # Visualizza il grafico della matrice di correlazione
            st.pyplot(fig)
        

            # Scatter plot umidità e temperatura
            st.subheader(' 📍 Scatter plot umidità e temperatura')
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(x=dataset_corrente['media umidità'], 
                            y=dataset_corrente['media temperatura'], 
                            ax=ax)
            ax.set_xlabel('Umidità')
            ax.set_ylabel('Temperatura')
            ax.set_title(f"Scatter plot ({scelta_dataset})")
            st.pyplot(fig)
        #else:
         #   st.error("Il dataset interpolato non è disponibile.")


            #Test Stazionarietà e autocorrelazione
            from statsmodels.tsa.stattools import adfuller
            import statsmodels.graphics.tsaplots as sgt
            from statsmodels.tsa.stattools import grangercausalitytests
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            st.subheader(' ⏳ Autocorrelazione e autocorrelazione parziale')
             # Grafico ACF
                    # Slider per il numero di lag dell'ACF
            lags = st.slider("Numero di lag per l'ACF", 1, 50, 24)

            fig, ax = plt.subplots(figsize=(12, 6))
            sgt.plot_acf(dataset_corrente['numero di insetti'], lags=lags, zero=False, ax=ax)
            ax.set_title('Autocorrelazione (ACF)')
            st.pyplot(fig)  # Uso di st.pyplot() per visualizzare il grafico
    
            # Grafico PACF
            fig, ax = plt.subplots(figsize=(12, 6))
            sgt.plot_pacf(dataset_corrente['numero di insetti'], lags=lags, zero=False, ax=ax)
            ax.set_title('Autocorrelazione Parziale (PACF)')
            st.pyplot(fig)

            #Decomposizione serie storica 
            from statsmodels.tsa.seasonal import seasonal_decompose
            st.subheader(' 🔍 Decomposizione della serie storica Numero di insetti')
            # Aggiunta di uno slider per scegliere il periodo per la decomposizione
            periodo = st.slider("Periodo della decomposizione", 1, 48, 24)
            decomposizione = seasonal_decompose(dataset_corrente['numero di insetti'], model='additive', period=periodo)  
            # Creo la figura per Streamlit
            fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
            decomposizione.observed.plot(ax=axes[0], color='blue', legend=True, title='Osservata')
            decomposizione.trend.plot(ax=axes[1], color='green', legend=True, title='Trend')
            decomposizione.seasonal.plot(ax=axes[2], color='red', legend=True, title='Stagionale')
            decomposizione.resid.plot(ax=axes[3], color='black', legend=True, title='Residuo')
            plt.tight_layout()
            st.pyplot(fig)  

    # Sezione Modelli
    elif scelta_funzione == 'Modelli':
        st.header(' 🧠 Modelli')
        st.write("✅ Regressione Lineare :")
        st.write("Questo modello è ideale per analizzare relazioni tra variabili continue e prevedere valori numerici futuri.")
        st.write("È stato scelto per la sua semplicità, interpretabilità e affidabilità nelle analisi predittive.")

        st.write("✅ Albero Decisionale (Classificazione) :")
        st.write("Gli alberi decisionali sono potenti strumenti di classificazione che permettono di separare i dati in categorie comprensibili.") 
        st.write("Abbiamo scelto questo metodo per la sua capacità di gestire dati complessi e fornire risultati facilmente interpretabili.")

        # Aggiungi un selectbox per scegliere il tipo di problema
        tipo_problema = st.selectbox("Seleziona il tipo di problema", ["Regressione", "Classificazione"])

        # Recupera il dataset lagged per la sezione Modelli
        dataset_lagged = datasets_lagged.get(scelta_dataset, None)

        if dataset_lagged is not None:
            if tipo_problema == "Regressione":
                # Sottosezione: Regressione Lineare
                with st.expander(' Regressione Lineare'):
                    st.subheader(f' 📚 Regressione Lineare ({scelta_dataset})')

                    # Parametri selezionabili per la regressione
                    max_lag = st.slider("Seleziona il numero massimo di lag", min_value=1, max_value=5, value=3)
                
                # Verifica se le colonne lagged esistono nel dataset lagged
                    if all([f'numero di insetti_lag_{i}' in dataset_lagged.columns for i in range(1, max_lag + 1)]):
                    # Preparazione dei dati
                        X = dataset_lagged[[f'numero di insetti_lag_{i}' for i in range(1, max_lag + 1)]]
                        y = dataset_lagged['numero di insetti']
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

                        # Addestramento del modello
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        r2 = model.score(X_test, y_test)
                        st.write(f"R² score sul dataset di test ({scelta_dataset}): {r2:.2f}")

                    # Predizioni
                        y_pred = model.predict(X_test)
                        results = X_test.copy()
                        results['Valori Reali'] = y_test.values
                        results['Valori Predetti'] = y_pred

                    # Mostra i risultati
                        st.dataframe(results)

                    # Grafico
                        fig, ax = plt.subplots(figsize=(14, 6))
                        ax.plot(results.index, results['Valori Reali'], label='Valori Reali', color='blue')
                        ax.plot(results.index, results['Valori Predetti'], label='Valori Predetti', color='red', linestyle='--')
                        ax.set_title(f'Regressione Lineare: Valori Reali vs Predetti ({scelta_dataset})')
                        ax.legend()
                        st.pyplot(fig)
                    else:
                        st.error("Le colonne lagged non sono presenti nel dataset!")

            else:
            # Sottosezione: Modello di classificazione - Albero Decisionale
                with st.expander("Modello di classificazione: Albero decisionale"):
                    st.subheader(f' 🌳 Albero decisionale ({scelta_dataset})')

                # Parametri selezionabili per l'albero decisionale
                    max_depth = st.slider("Seleziona la profondità massima dell'albero", min_value=1, max_value=10, value=3)
                    min_samples_split = st.slider("Seleziona il numero minimo di campioni per dividere un nodo", min_value=2, max_value=10, value=2)

                # Verifica se le colonne lagged esistono nel dataset lagged
                    if all([f'nuove catture (per evento)_lag_{i}' in dataset_lagged.columns for i in range(1, max_lag + 1)]):
                    # Preparazione dei dati
                        X = dataset_lagged[[f'nuove catture (per evento)_lag_{i}' for i in range(1, max_lag + 1)]]
                        y = dataset_lagged['nuove catture (per evento)']

                    # Divisione dei dati in training e test set
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

                    # Addestramento del modello con i parametri selezionati
                        from sklearn.tree import DecisionTreeClassifier, plot_tree
                        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
                        model.fit(X_train, y_train)
                        accuracy = model.score(X_test, y_test)
                        st.write(f"Accuratezza sul dataset di test ({scelta_dataset}): {accuracy:.2f}")

                    # Predizioni
                        y_pred = model.predict(X_test)
                        results = X_test.copy()
                        results['Valori Reali'] = y_test.values
                        results['Valori Predetti'] = y_pred

                    # Mostra i risultati
                        st.dataframe(results)

                    # Grafico: visualizzazione dell'albero decisionale
                        fig, ax = plt.subplots(figsize=(16, 10))
                        plot_tree(model, feature_names=X.columns, class_names=['0', '1', '2'], filled=True, ax=ax)
                        plt.title("Visualizzazione dell'Albero Decisionale")
                        st.pyplot(fig)
                    else:
                        st.error("Le colonne lagged non sono presenti nel dataset!")


            
