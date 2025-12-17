import streamlit as st
import pandas as pd
import joblib
import mysql.connector
from mysql.connector import Error
import datetime
import numpy as np

# ============================================
# 1. CONFIGURAÇÕES E CREDENCIAIS (GCP MYSQL)
# ============================================
# ⚠️ ATENÇÃO: Substitua pelos dados reais do seu Google Cloud Platform
DB_CONFIG = {
    'host': '34.151.213.212',       # COLOQUE AQUI SEU IP PÚBLICO DA INSTÂNCIA GCP
    'database': 'dados-pacientes',       # COLOQUE O NOME DO BANCO QUE VOCÊ CRIOU
    'user': 'admin',               # USUÁRIO (Geralmente 'root')
    'password': 'N8!pZ7@wQ3#rL9$s'  # A SENHA QUE VOCÊ DEFINIU NO GCP
}

# Configuração da Página
st.set_page_config(page_title="Predição Diabetes AV4", page_icon="🏥", layout="centered")

# ============================================
# 2. FUNÇÕES DE BANCO DE DADOS (MySQL)
# ============================================
def get_db_connection():
    """Tenta estabelecer conexão com o banco"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            return conn
    except Error as e:
        st.error(f"❌ Erro de conexão com o Banco de Dados: {e}")
        return None

def init_db():
    """Cria a tabela de logs se ela não existir"""
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            # Criação da tabela compatível com MySQL
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    inputs TEXT,
                    predicao VARCHAR(50),
                    certeza FLOAT,
                    data_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            cursor.close()
            conn.close()
        except Error as e:
            st.error(f"Erro ao criar tabela: {e}")

def salvar_interacao(inputs_dict, resultado, certeza):
    """Salva o diagnóstico no banco da nuvem"""
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            inputs_str = str(inputs_dict)
            query = "INSERT INTO logs (inputs, predicao, certeza) VALUES (%s, %s, %s)"
            valores = (inputs_str, resultado, float(certeza))
            
            cursor.execute(query, valores)
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except Error as e:
            st.error(f"Erro ao salvar registro: {e}")
            return False

# Inicializa a estrutura do banco ao abrir o app
init_db()

# ============================================
# 3. CARREGAMENTO DA INTELIGÊNCIA ARTIFICIAL
# ============================================
@st.cache_resource
def carregar_modelo():
    try:
        # Tenta carregar o arquivo completo (Modelo + Scaler + Imputer)
        dados = joblib.load('modelo_diabetes_completo.pkl')
        return dados['modelo'], dados['scaler'], dados['imputer']
    except FileNotFoundError:
        st.error("⚠️ Arquivo 'modelo_diabetes_completo.pkl' não encontrado!")
        st.warning("Por favor, faça o upload do arquivo gerado no Colab para a mesma pasta deste script.")
        return None, None, None
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, None, None

model, scaler, imputer = carregar_modelo()

# ============================================
# 4. INTERFACE DO USUÁRIO (FRONTEND)
# ============================================
st.title("🏥 Sistema de Diagnóstico - Diabetes")
st.markdown("**Projeto AV4** | Integração Machine Learning & Cloud Computing")
st.markdown("---")

if model is not None:
    st.sidebar.header("📝 Prontuário do Paciente")
    
    # Inputs organizados em 2 colunas para melhor visualização
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Gravidezes", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glicose (mg/dL)", min_value=0, max_value=300, value=120, help="Nível de glicose no sangue")
        blood_pressure = st.number_input("Pressão Sanguínea (mmHg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Espessura da Pele (mm)", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("Insulina (mu U/ml)", min_value=0, max_value=900, value=0)
        bmi = st.number_input("IMC", min_value=0.0, max_value=70.0, value=32.0, format="%.1f")
        dpf = st.number_input("Histórico Familiar (Pedigree)", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
        age = st.number_input("Idade", min_value=0, max_value=120, value=33)

    st.markdown("---")
    
    # Botão de Ação
    if st.button("🔍 Processar Diagnóstico", use_container_width=True):
        with st.spinner('A Inteligência Artificial está analisando os dados...'):
            try:
                # 1. Organizar dados (A ordem DEVE ser a mesma do treinamento)
                features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
                features_array = np.array(features).reshape(1, -1)
                
                # 2. Aplicar Imputação (Preencher zeros/nulos se necessário)
                features_imputed = imputer.transform(features_array)
                
                # 3. Aplicar Normalização (Deixar na mesma escala do treino)
                features_scaled = scaler.transform(features_imputed)
                
                # 4. Predição
                prediction = model.predict(features_scaled)
                proba = model.predict_proba(features_scaled)
                
                # Lógica de Exibição
                classe = prediction[0]
                confianca = proba[0][classe] * 100
                resultado_texto = "DIABÉTICO" if classe == 1 else "SAUDÁVEL (Não Diabético)"
                
                # 5. Mostrar Resultado na Tela
                if classe == 1:
                    st.error(f"## 🚨 Resultado: {resultado_texto}")
                    st.write(f"**Probabilidade calculada:** {confianca:.2f}% de certeza.")
                else:
                    st.success(f"## ✅ Resultado: {resultado_texto}")
                    st.write(f"**Probabilidade calculada:** {confianca:.2f}% de certeza.")

                # 6. Salvar no MySQL GCP
                dados_log = {
                    'Glicose': glucose,
                    'IMC': bmi,
                    'Idade': age,
                    'BP': blood_pressure
                }
                
                sucesso_db = salvar_interacao(dados_log, resultado_texto, confianca)
                if sucesso_db:
                    st.toast("✅ Registro salvo na nuvem com sucesso!", icon="☁️")
                
            except Exception as e:
                st.error(f"Ocorreu um erro no processamento: {e}")

# ============================================
# 5. ÁREA ADMINISTRATIVA (VISUALIZAR BANCO)
# ============================================
st.divider()
st.subheader("🔐 Área Administrativa")

if st.checkbox("Visualizar Banco de Dados (Google Cloud SQL)"):
    conn = get_db_connection()
    if conn:
        try:
            query = "SELECT * FROM logs ORDER BY data_hora DESC LIMIT 50"
            df_logs = pd.read_sql(query, conn)
            
            if not df_logs.empty:
                st.dataframe(df_logs, use_container_width=True)
                st.info(f"Mostrando os últimos {len(df_logs)} registros da nuvem.")
            else:
                st.warning("Nenhum registro encontrado no banco de dados.")
            
            conn.close()
        except Error as e:
            st.error(f"Erro ao buscar dados: {e}")