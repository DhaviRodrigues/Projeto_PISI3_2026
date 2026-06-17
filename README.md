
---
title: PISI3 Dashboard
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
# 🚀 Como rodar o projeto

Siga o passo a passo abaixo para executar o projeto corretamente.

---

## 📥 1. Clonar o repositório

```bash
git clone https://github.com/DhaviRodrigues/Projeto_PISI3_2026.git
cd seu-repositorio
```

---

## 🐍 2. Criar e ativar ambiente virtual

### Criar:

```bash
python -m venv venv
```

### Ativar:

* **Windows**

```bash
venv\Scripts\activate
```

* **Linux / Mac**

```bash
source venv/bin/activate
```

---

## 📦 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## 📊 4. Baixar o dataset

👉 Link do Kaggle:
https://www.kaggle.com/datasets/anandshaw2001/imdb-data

### Passos:

1. Baixe o dataset
2. Coloque o arquivo dentro do repositório (na raiz ou onde seu notebook espera)

---

## 🧹 5. Gerar o arquivo `.parquet`

A limpeza dos dados é feita via notebook.

### Caminho:

```bash
limpeza_de_dados/conversao_parquet.ipynb
```

### O que fazer:

1. Abra o notebook:

```bash
jupyter notebook
```

2. Navegue até:

```
limpeza_de_dados/conversao_parquet.ipynb
```

3. Execute **a célula responsável pela conversão** (ou todas, se preferir)

👉 Isso irá gerar um arquivo `.parquet`

---

## 📁 6. Mover o arquivo `.parquet`

Após gerar o arquivo:

➡️ Mova o `.parquet` para a pasta:

```bash
dashboard/
```

---

## ▶️ 7. Executar o dashboard

Entre na pasta:

```bash
cd dashboard
```

E rode:

```bash
python dashboard.py
```

---

## ❗ Possíveis erros comuns

### Arquivo `.parquet` não encontrado

➡️ Verifique se você moveu o arquivo corretamente para a pasta `dashboard/`

---

### Notebook não roda

➡️ Certifique-se de que instalou o Jupyter:

```bash
pip install jupyter
```

---

### Dependências faltando

➡️ Rode novamente:

```bash
pip install -r requirements.txt
```

---

## ✅ Fluxo resumido

1. Clonar repo
2. Instalar dependências
3. Baixar dataset
4. Rodar notebook (`conversao_parquet.ipynb`)
5. Gerar `.parquet`
6. Mover para `dashboard/`
7. Rodar `dashboard.py`

---

Pronto! 🎉
