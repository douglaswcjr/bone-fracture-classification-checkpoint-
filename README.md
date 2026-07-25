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

O projeto segue as três entregas propostas no checkpoint — exploração/pré-
processamento, construção/avaliação de modelos, e entrega da solução — e as
seções abaixo documentam, para cada uma, **a pergunta que estava sendo
respondida, as técnicas usadas e os insights que realmente apareceram nos
dados**, não só o "como rodar".

## Raciocínio do projeto, entrega por entrega

### Parte 1 — Exploração e pré-processamento (`notebooks/01_exploracao.ipynb`)

**Perguntas que essa etapa precisava responder:** como os dados estão
organizados e o que preciso fazer para poder classificar imagens inteiras com
eles? Qual o tamanho real do dataset e ele está balanceado? O que vai ser mais
difícil na hora de classificar?

**Técnicas usadas:**
- Parsing das anotações YOLO (`src/dataset.py::parse_yolo_label`) e derivação
  de um rótulo por imagem via **classe majoritária entre as boxes** (empate
  resolvido pelo menor `class_id` — regra determinística).
- Análise de distribuição de classes por split (contagens e gráficos de barra).
- Análise dos **nomes de arquivo** (`<id-base>.rf.<hash>.jpg`) para detectar
  quantas imagens são, na verdade, cópias aumentadas da mesma foto original.
- Grids de exemplos por classe com as bounding boxes desenhadas
  (`matplotlib.patches`), incluindo casos de imagens com mais de uma classe
  anotada.
- Pipeline de pré-processamento (`tf.data`): decodificação JPEG e resize
  640×224 → 224×224 (padrão MobileNetV2), normalização deixada para dentro do
  modelo (Parte 2).

**Principais insights (o que mudou o rumo do projeto):**
1. **O dataset "de 1539 imagens" na verdade tem só 244 raios-X originais
   únicos** (~6,3 cópias aumentadas cada, geradas pelo próprio Roboflow antes
   do dataset ser dividido em train/valid/test).
2. **Isso causa vazamento entre splits**: 70 imagens-base aparecem em treino
   *e* validação, 47 em treino *e* teste, 19 em validação *e* teste. Ou seja,
   o modelo pode treinar numa versão espelhada/rotacionada de uma foto e ser
   "testado" numa cópia quase idêntica dela.
3. **Corrigir isso com um re-split agrupado por imagem-base não é viável**:
   ao nível de imagem original, a classe *Segmental* tem só 2 exemplares e
   *Linear* só 4 — não dá para garantir as três partições sem zerar alguma
   classe rara. Decisão tomada: manter o split oficial do Roboflow e
   **documentar o vazamento como limitação**, em vez de fingir que ele não
   existe.
4. **Desbalanceamento severo**: no treino, a classe mais comum
   (*Transverse Displaced*, 537 imagens) é dezenas de vezes mais frequente
   que a mais rara (*Segmental*, 12 imagens). *Linear* e *Segmental* também
   não têm **nenhuma** imagem no split de teste oficial.
5. **Hipótese levantada para testar na Parte 2**: pares de classes que
   descrevem o mesmo tipo de fratura com/sem desvio (*Transverse* vs.
   *Transverse Displaced*, *Oblique* vs. *Oblique Displaced*) deveriam ser os
   mais confundidos pelo modelo, por serem visualmente parecidos.

### Parte 2 — Construção e avaliação dos modelos (`notebooks/02_modelagem.ipynb`)

**Perguntas que essa etapa precisava responder:** treinar do zero ou
aproveitar transfer learning, dado o tamanho pequeno do dataset? Quais
métricas fazem sentido com esse desbalanceamento? A hipótese de classes
parecidas da Parte 1 se confirma? O modelo está de fato olhando para a
fratura, ou para outra coisa na imagem?

**Técnicas usadas:**
- **CNN convolucional do zero** (`src/models.py::build_cnn_scratch`): 4 blocos
  Conv2D+BatchNorm+MaxPooling, GlobalAveragePooling, Dense+Dropout.
- **Transfer learning com MobileNetV2** (`build_mobilenetv2`): base ImageNet
  congelada + treino do head, depois **fine-tuning** da base inteira com
  learning rate baixo (1e-5).
- Regularização igual para os dois modelos: data augmentation embutido no
  modelo (flip, rotação, zoom, brilho — ativo só em treino), Dropout,
  `class_weight` balanceado (`sklearn.utils.class_weight`) para compensar o
  desbalanceamento da Parte 1, `EarlyStopping` e `ReduceLROnPlateau`.
- Avaliação (`src/evaluate.py`): `classification_report` por classe, matriz
  de confusão e curvas de aprendizado (loss/accuracy treino vs. validação).
- Interpretabilidade: **Grad-CAM** implementado manualmente com
  `tf.GradientTape` (`src/gradcam.py`), aplicado a acertos e erros do melhor
  modelo.

**Principais insights (resultado real do treino, não estimado):**

| Modelo | Accuracy (teste) | F1 macro (teste) |
| --- | --- | --- |
| CNN do zero | 0.016 | 0.003 |
| MobileNetV2 (transfer learning) | 0.563 | 0.467 |

1. **A CNN do zero colapsou** — o `val_loss` piora a cada época após a
   primeira e o modelo praticamente só prevê uma única classe no teste. Com
   ~1.300 imagens espalhadas por 10 classes desbalanceadas (uma delas com
   peso 22× maior que outra no `class_weight`), não há dado suficiente para
   uma rede aleatória aprender features úteis de forma estável — é o
   argumento prático mais forte a favor de transfer learning neste projeto,
   não uma falha de implementação (a mesma arquitetura de treino funcionou
   bem no MobileNetV2).
2. **MobileNetV2 generalizou bem melhor** (accuracy 56%, macro F1 0.47),
   reaproveitando features gerais já aprendidas na ImageNet.
3. **A hipótese da Parte 1 só se confirmou parcialmente.** Na matriz de
   confusão, *Transverse* e *Oblique* tiveram 100% de recall — nenhuma
   confusão com suas variantes "Displaced". Quem concentrou os erros foi
   *Transverse Displaced* (a classe mais frequente do treino): só 5 de 22
   exemplos reais foram classificados corretamente, com os erros
   **espalhados** por quase todas as outras classes, não concentrados na sua
   "contraparte" visual. Ou seja, a classe mais difícil não foi a prevista
   por semelhança visual, e sim uma classe com anotações provavelmente mais
   heterogêneas (fraturas em posições/ângulos bem variados sob o mesmo
   rótulo).
4. **As métricas acima devem ser lidas com a ressalva do vazamento** (Parte
   1): o desempenho em raios-X de pacientes realmente novos tende a ser
   inferior ao medido aqui.

### Parte 3 — Entrega da solução (`app.py`, `src/predicao.py`, `src/treino.py`)

**Perguntas que essa etapa precisava responder:** como entregar isso de forma
utilizável por alguém não-técnico? Como organizar o código para separar
treino de inferência e permitir reproduzir o pipeline inteiro fora do
notebook?

**Técnicas usadas:**
- App **Streamlit** (`app.py`): upload de imagem → classificação →
  probabilidades por classe → overlay Grad-CAM, tudo reaproveitando as
  mesmas funções de pré-processamento/predição usadas no treino
  (`predict_bytes`), com o modelo carregado uma única vez via
  `st.cache_resource`.
- Separação de scripts auxiliares: `src/treino.py` (CLI de treino,
  reproduz os notebooks fora do Jupyter) e `src/predicao.py` (inferência,
  usada tanto pela CLI quanto pelo app).

**Principal insight (um bug real, encontrado só ao testar de ponta a
ponta):** o Grad-CAM funcionava perfeitamente nos testes feitos com o modelo
recém-treinado em memória, mas **quebrava no app** — que carrega o modelo
salvo em disco (`tf.keras.models.load_model`). A causa: o Grad-CAM dependia
de atributos Python customizados (`model.base_model`) para localizar a base
MobileNetV2 aninhada, e **esses atributos não sobrevivem a um
`model.save()`/`load_model()`** do Keras — só existiam no objeto em memória.
A correção (`src/gradcam.py::_find_nested_model_layer`) passou a localizar a
base aninhada **estruturalmente** (procurando, dentro de `model.layers`, o
submodelo que contém camadas convolucionais), o que funciona tanto em
memória quanto após salvar/recarregar. Esse achado reforça por que vale a
pena testar o fluxo completo (treinar → salvar → recarregar → servir) e não
só o código "quente" dentro do notebook.

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
> majoritária entre as boxes anotadas — a regra completa e suas implicações
> estão documentadas na seção "Parte 1" acima e no notebook correspondente.

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
