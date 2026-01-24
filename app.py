from curses import wrapper

import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import  matplotlib.pyplot as plt
import seaborn as sns
from model.preprocess import split_features_target, scale_height_weight
from model.registry.registry import MODEL_REGISTRY

st.set_page_config(page_title="ML Assignmet 2 - Obesity", layout="wide")
st.title("Obesity Level Classification - ML Assignment 2")

st.markdown("""
Upload the obesity dataset CSV (or test subset). Select a model to train and evaluate.
""")


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
model_selected = st.selectbox("Select Model", list(MODEL_REGISTRY.keys()))


test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)


if uploaded_file:
    df = pd.read_csv(uploaded_file)

    X, y = split_features_target(df, target_col="NObeyesdad")

    X = scale_height_weight(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y
    )


    wrapper_cls = MODEL_REGISTRY[model_selected]
    wrapper = wrapper_cls()

    st.subheader(f"Training: {wrapper.name}")
    wrapper.fit(X_train, y_train)

    metrics = wrapper.evaluate(X_test, y_test)

    y_pred = wrapper.predict(X_test)

    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("Evaluation Metrics ")
        metrics_df = pd.DataFrame([metrics])
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))


    with col2:
        st.subheader("Confusion Matrx")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)


















































































