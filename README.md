# 🚀 Como rodar o projeto

Siga os passos abaixo para configurar e executar o projeto corretamente.

---

## 📥 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

---

## 🐍 2. Criar e ativar ambiente virtual

### Criar ambiente:

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

## 📊 4. Baixar e configurar o dataset

Este projeto utiliza um dataset externo, que deve ser baixado manualmente.

👉 Link do Kaggle:
https://www.kaggle.com/SEU-LINK-AQUI

### Passos:

1. Acesse o link acima
2. Baixe o dataset
3. Extraia os arquivos (se estiverem compactados)
4. Coloque os arquivos dentro da pasta:

```bash
data/
```

⚠️ **Importante:**
Certifique-se de que os arquivos estão diretamente dentro da pasta `data` e não em subpastas inesperadas.

---

## 🧹 5. Pré-processamento dos dados

Antes de rodar o projeto principal, execute a etapa de limpeza:

```bash
python src/preprocessing.py
```

Essa etapa irá:

* Tratar valores nulos
* Corrigir inconsistências
* Preparar os dados para uso

---

## ▶️ 6. Executar o projeto

Após preparar os dados, rode:

```bash
python src/main.py
```

---

## 🧪 (Opcional) Rodar com Jupyter Notebook

Se quiser explorar os dados:

```bash
jupyter notebook
```

Abra os arquivos dentro da pasta `notebooks/`.

---

## ❗ Possíveis erros comuns

### Erro: módulo não encontrado

➡️ Certifique-se de que o ambiente virtual está ativado

---

### Erro: arquivo não encontrado

➡️ Verifique se o dataset está na pasta `data/`

---

### Erro de dependências

➡️ Rode novamente:

```bash
pip install -r requirements.txt
```

---

## ✅ Pronto!

Se tudo deu certo, o projeto deve rodar normalmente 🎉
