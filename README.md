# Classificação de Fraturas Ósseas em Raio-X

Checkpoint de Ciência de Dados (Alura) — protótipo de IA para apoiar a triagem
radiológica de fraturas ósseas em exames de raio-X, feito para um cenário de
hospital universitário fictício.

> ⚠️ **Projeto educacional.** Não é um dispositivo médico e não deve ser usado
> para diagnóstico real.

## Objetivo

Treinar um classificador de imagens capaz de identificar o tipo de fratura
óssea (ou ausência de fratura) em um raio-X, a partir do dataset **HBFMID**
(Human Bone Fractures Multi-modal Image Dataset), e entregar isso como uma
aplicação simples de upload + predição, com interpretabilidade via Grad-CAM.

10 classes: `Comminuted`, `Greenstick`, `Healthy`, `Linear`, `Oblique`,
`Oblique Displaced`, `Segmental`, `Spiral`, `Transverse`, `Transverse Displaced`.

## Estrutura do repositório

```
├── data/                   # dataset (não versionado — ver "Dataset" abaixo)
├── notebooks/
│   ├── 01_exploracao.ipynb # Parte 1: exploração e pré-processamento
│   └── 02_modelagem.ipynb  # Parte 2: treino, avaliação e Grad-CAM dos modelos
├── src/
│   ├── dataset.py          # rótulo por imagem a partir das boxes YOLO + pipeline tf.data
│   ├── models.py            # arquiteturas (CNN do zero e MobileNetV2)
│   ├── evaluate.py          # métricas, matriz de confusão, curvas de aprendizado
│   ├── gradcam.py           # Grad-CAM (tf.GradientTape)
│   ├── treino.py            # CLI de treino: `python -m src.treino`
│   └── predicao.py          # inferência: `python -m src.predicao`
├── app.py                   # app Streamlit (upload → classificação + Grad-CAM)
├── models/                  # modelos treinados (.keras) + class_names.json
├── reports/figures/         # figuras geradas pelo treino/avaliação
└── requirements.txt
```

## Dataset

O dataset **HBFMID** está disponível no Kaggle:
<https://www.kaggle.com/datasets/jockeroika/human-bone-fractures-image-dataset>
(anotações no formato YOLOv8, 1539 imagens 640×640, já com data augmentation
aplicado pelo Roboflow — 3 versões por imagem original).

Ele não é versionado neste repositório (arquivo grande, redistribuível pela
fonte original). Para reproduzir:

1. Baixe o `.zip` do Kaggle (ou do link do Roboflow no `data.yaml`).
2. Extraia de forma que a estrutura final fique:

```
data/
├── data.yaml
├── train/{images,labels}
├── valid/{images,labels}
└── test/{images,labels}
```

> **Nota de pré-processamento:** o dataset original tem anotações de
> *detecção* (bounding boxes). Como o projeto pede classificação da imagem
> inteira, `src/dataset.py` deriva um rótulo por imagem a partir da classe
> majoritária entre as boxes anotadas — a regra completa está documentada no
> notebook da Parte 1, junto com uma limitação importante encontrada nos
> dados (vazamento de imagens aumentadas entre os splits train/valid/test).

## Instalação

Usa [`uv`](https://docs.astral.sh/uv/) para gerenciar o ambiente Python.

```bash
uv venv
uv pip install -r requirements.txt
```

Ative o ambiente (Windows / Git Bash: `source .venv/Scripts/activate`; Linux/macOS:
`source .venv/bin/activate`) antes dos comandos abaixo, ou prefixe cada comando
com `.venv/Scripts/python.exe -m` (Windows) / `.venv/bin/python -m` (Linux/macOS).

Para rodar os notebooks, registre o ambiente como kernel do Jupyter:

```bash
python -m ipykernel install --user --name hbfmid-venv --display-name "Python (.venv HBFMID)"
```

## Como rodar

**Notebooks** (Parte 1 e Parte 2, nessa ordem — selecione o kernel "Python (.venv HBFMID)"):

```bash
jupyter lab
```

**Treino via linha de comando** (reproduz o que os notebooks fazem, gera
`models/*.keras` e figuras em `reports/figures/`):

```bash
python -m src.treino --model both
```

**Classificar uma imagem via linha de comando:**

```bash
python -m src.predicao models/mobilenetv2_transfer.keras caminho/para/imagem.jpg
```

**App Streamlit** (upload de imagem → classificação + Grad-CAM):

```bash
streamlit run app.py
```

## Pipeline

1. **Pré-processamento** (`src/dataset.py`, Parte 1): rótulo por imagem a
   partir das boxes YOLO, decodificação/resize 640→224, split oficial
   train/valid/test do Roboflow.
2. **Treino** (`src/models.py` + `src/treino.py`, Parte 2): CNN do zero e
   MobileNetV2 (transfer learning + fine-tuning), com data augmentation,
   dropout, `class_weight` balanceado, early stopping e redução de learning
   rate.
3. **Avaliação** (`src/evaluate.py`, Parte 2): classification report, matriz
   de confusão e curvas de aprendizado para os dois modelos.
4. **Interpretabilidade** (`src/gradcam.py`, Parte 2 e app): Grad-CAM sobre a
   última camada convolucional, mostrando as regiões da imagem que mais
   pesaram na predição.
5. **Entrega** (`app.py`, Parte 3): app Streamlit para classificar novas
   imagens com o modelo salvo.

## Resultados

Ver `notebooks/02_modelagem.ipynb` para o comparativo completo (métricas,
matrizes de confusão, curvas de aprendizado e Grad-CAM). Resumo:

| Modelo | Accuracy (teste) | F1 macro (teste) |
| --- | --- | --- |
| CNN do zero | 0.016 | 0.003 |
| MobileNetV2 (transfer learning) | 0.563 | 0.467 |

A CNN treinada do zero **colapsou** (praticamente só prevê uma classe) — resultado
esperado ao treinar uma rede do zero em ~1.300 imagens espalhadas por 10 classes
desbalanceadas, e um bom argumento prático a favor de transfer learning em datasets
pequenos. O MobileNetV2 generalizou bem melhor. Discussão completa (incluindo por
que a hipótese inicial sobre "classes parecidas" só se confirmou parcialmente) na
seção 5 do `02_modelagem.ipynb`.

**Limitação importante:** o split de teste do Roboflow compartilha imagens
*originais* (antes do data augmentation) com o treino — o dataset de 1539
imagens vem de pouco mais de 200 raios-X únicos. Isso tende a inflar as
métricas acima em relação ao desempenho esperado em pacientes/exames
realmente novos. Detalhes e números exatos na Parte 1 do notebook.
