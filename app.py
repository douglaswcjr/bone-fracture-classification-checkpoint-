"""App Streamlit: upload de uma imagem de raio-X, classificação do tipo de
fratura e visualização Grad-CAM para interpretabilidade.

Rodar com: streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dataset import load_class_names
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.predicao import load_model, predict_bytes

MODELS_DIR = Path("models")
st.set_page_config(page_title="Classificador de Fraturas Ósseas", page_icon="🦴", layout="centered")


@st.cache_resource
def _load_model(model_path: str):
    return load_model(model_path)


@st.cache_resource
def _load_class_names(path: str = "models/class_names.json"):
    return load_class_names(path)


def main() -> None:
    st.title("🦴 Classificador de Fraturas Ósseas")
    st.caption(
        "Protótipo educacional para apoio à triagem radiológica — "
        "**não substitui avaliação médica.**"
    )

    available_models = sorted(MODELS_DIR.glob("*.keras"))
    if not available_models:
        st.error(
            "Nenhum modelo treinado encontrado em `models/`. Rode primeiro:\n\n"
            "`python -m src.treino --model both`"
        )
        return

    model_path = st.selectbox(
        "Modelo", available_models, format_func=lambda p: p.stem, index=len(available_models) - 1
    )
    model = _load_model(str(model_path))
    class_names = _load_class_names()

    uploaded_file = st.file_uploader("Envie uma imagem de raio-X (JPG ou PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Envie uma imagem para ver a classificação prevista.")
        return

    image_bytes = uploaded_file.getvalue()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagem enviada")
        st.image(image_bytes, use_container_width=True)

    with st.spinner("Classificando..."):
        predicted_class, probabilities, batch = predict_bytes(image_bytes, model, class_names)
        heatmap, _ = make_gradcam_heatmap(batch, model)
        overlay = overlay_heatmap(batch[0].astype("uint8"), heatmap)

    with col2:
        st.subheader("Regiões usadas pelo modelo (Grad-CAM)")
        st.image(overlay, use_container_width=True)

    st.success(f"Classe prevista: **{predicted_class}**")

    probs_series = pd.Series(probabilities, name="probabilidade").sort_values(ascending=False)
    st.bar_chart(probs_series)


if __name__ == "__main__":
    main()
