import streamlit as st
import pandas as pd
import joblib
import sqlite3
import datetime
import numpy as np

# ============================================
# 1. CONFIGURAÇÃO DO BANCO DE DADOS (SQLite)
# ============================================
def init_db():
    conn = sqlite3.connect('interacoes.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inputs TEXT,
            predicao TEXT,
            data_hora TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def salvar_interacao(inputs_dict, resultado):
    conn = sqlite3.connect('interacoes.db')
    c = conn.cursor()
    # Converte o dicionário de inputs para string para salvar no banco
    inputs_str = str(inputs_dict)
    c.execute("INSERT INTO logs (inputs, predicao, data_hora) VALUES (?, ?, ?)",
              (inputs_str, resultado, datetime.datetime.now()))
    conn.commit()
    conn.close()

# Inicia o banco ao abrir
init_db()

# ============================================
# 2. CARREGAMENTO DO MODELO E PROCESSADORES
# ============================================
@st.cache_resource
def carregar_modelo():
    try:
        # Carrega o dicionário com tudo que salvamos no Colab
        dados = joblib.load('modelo_diabetes_completo.pkl')
        return dados['modelo'], dados['scaler'], dados['imputer']
    except FileNotFoundError:
        return None, None, None

model, scaler, imputer = carregar_modelo()

# ============================================
# 3. INTERFACE DO USUÁRIO (STREAMLIT)
# ============================================
st.set_page_config(page_title="Predição Diabetes AV4", page_icon="🏥")

st.title("🏥 Sistema de Auxílio Diagnóstico - Diabetes")
st.markdown("Projeto AV4 - Lucas Pereira | Baseado no dataset Pima Indians")

if model is None:
    st.error("ERRO CRÍTICO: Arquivo 'modelo_diabetes_completo.pkl' não encontrado. Faça o upload dele para o GitHub.")
else:
    st.sidebar.header("Dados do Paciente")
    
    # Formulário com as 8 features exatas do dataset
    # Usando colunas para ficar mais bonito
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Nº de Gravidezes", 0, 20, 1)
        glucose = st.number_input("Glicose (mg/dL)", 0, 300, 120)
        blood_pressure = st.number_input("Pressão Sanguínea (mm Hg)", 0, 200, 70)
        skin_thickness = st.number_input("Espessura da Pele (mm)", 0, 100, 20)
    
    with col2:
        insulin = st.number_input("Insulina (mu U/ml)", 0, 900, 79)
        bmi = st.number_input("IMC (Índice de Massa Corporal)", 0.0, 70.0, 32.0)
        dpf = st.number_input("Histórico Familiar (Pedigree 0-3)", 0.0, 3.0, 0.5)
        age = st.number_input("Idade", 0, 120, 33)

    # Botão de Ação
    if st.button("Realizar Diagnóstico com IA"):
        # 1. Organizar os dados na mesma ordem do treinamento
        features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        features_array = np.array(features).reshape(1, -1)
        
        # 2. Aplicar o Imputer (caso tenha zeros que precisem ser média, embora o input não permita zero onde não deve)
        # Nota: O imputer espera 8 colunas.
        features_imputed = imputer.transform(features_array)
        
        # 3. Aplicar o Scaler (Normalização) - O PASSO MAIS IMPORTANTE
        features_scaled = scaler.transform(features_imputed)
        
        # 4. Fazer a Predição
        prediction = model.predict(features_scaled)
        probabilidade = model.predict_proba(features_scaled)
        
        # Lógica de Resultado
        resultado_texto = "DIABÉTICO" if prediction[0] == 1 else "NÃO DIABÉTICO"
        prob_percent = probabilidade[0][prediction[0]] * 100
        
        # 5. Exibir Resultado
        st.divider()
        if prediction[0] == 1:
            st.error(f"### Resultado: {resultado_texto}")
            st.warning(f"O modelo tem {prob_percent:.1f}% de certeza.")
        else:
            st.success(f"### Resultado: {resultado_texto}")
            st.info(f"O modelo tem {prob_percent:.1f}% de certeza.")
            
        # 6. Salvar no Banco
        dados_input_log = {
            'Glicose': glucose,
            'IMC': bmi,
            'Idade': age,
            'Resultado': resultado_texto
        }
        salvar_interacao(dados_input_log, resultado_texto)
        st.toast("✅ Registro salvo no banco de dados!")

# ============================================
# 4. ÁREA ADMINISTRATIVA (Visualizar Banco)
# ============================================
st.divider()
if st.checkbox("Mostrar Histórico de Diagnósticos (Admin)"):
    conn = sqlite3.connect('interacoes.db')
    try:
        df_logs = pd.read_sql_query("SELECT * FROM logs ORDER BY data_hora DESC", conn)
        st.dataframe(df_logs)
    except:
        st.write("Ainda não há registros.")
    conn.close()