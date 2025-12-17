# 🏥 Sistema de Auxílio Diagnóstico - Diabetes (Projeto AV4)

> Projeto desenvolvido para a disciplina de Inteligência artificial  do curso de Análise e Desenvolvimento de Sistemas (IFPE - Campus Jaboatão dos Guararapes).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)
![GCP](https://img.shields.io/badge/Cloud-Google%20Cloud%20SQL-orange)
![Projeto](https://diabetes-prediction-model-kqqcfxdl4vzxjo5kxhyqeh.streamlit.app/)

## 📄 Sobre o Projeto

Este projeto consiste em uma aplicação web Full-Stack de **Data Science** voltada para a área da saúde. O sistema utiliza algoritmos de Machine Learning para prever a probabilidade de um paciente possuir Diabetes Mellitus com base em dados clínicos (Pima Indians Diabetes Database).

O diferencial desta aplicação é a integração completa entre um modelo preditivo de alta performance (**XGBoost**) e uma arquitetura em nuvem (**Google Cloud Platform**), onde cada diagnóstico realizado é registrado em um banco de dados MySQL remoto para auditoria e análise futura.

### 🎯 Objetivos
1.  Replicar e aprimorar experimentos científicos da literatura sobre detecção de diabetes.
2.  Disponibilizar o modelo através de uma interface amigável para profissionais de saúde.
3.  Garantir a persistência dos dados de diagnóstico em nuvem.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3
* **Frontend/Framework:** [Streamlit](https://streamlit.io/)
* **Machine Learning:**
    * XGBoost (Modelo Principal)
    * Scikit-Learn (Pré-processamento: KNN Imputer, StandardScaler)
    * Pandas & NumPy (Manipulação de dados)
* **Banco de Dados:** MySQL (Hospedado no Google Cloud SQL)
* **Persistência de Modelo:** Joblib

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
* Python 3.8 ou superior instalado.
* Conexão com a internet (para acessar o banco de dados no GCP).

### Passo a Passo

1.  **Clone o repositório**
    ```bash
    git clone [https://github.com/lucaspereira1dev/diabetes-prediction-model.git](https://github.com/lucaspereira1dev/diabetes-prediction-model.git)
    cd diabetes-prediciton-model
    ```

2.  **Crie um ambiente virtual (Opcional, mas recomendado)**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Instale as dependências**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuração do Banco de Dados**
    * Abra o arquivo `app.py`.
    * Localize a variável `DB_CONFIG` (linhas 13-18).
    * Insira as credenciais do seu banco MySQL (Host, User, Password, Database).
    * *Nota: O sistema criará a tabela `logs` automaticamente se ela não existir.*

5.  **Verifique o Modelo**
    * Certifique-se de que o arquivo `modelo_diabetes_completo.pkl` está na raiz do projeto. Este arquivo contém o modelo treinado e os objetos de pré-processamento.

6.  **Execute a aplicação**
    ```bash
    streamlit run app.py
    ```

7.  **Acesse no navegador**
    * O sistema abrirá automaticamente em: `http://localhost:8501`

---