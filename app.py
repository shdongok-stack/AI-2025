# Импорт библиотек
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем атрибуты из pickle
with open("HW1.pkl", "rb") as f:
    atr = pickle.load(f)

# Извлекаем данные из pickl
model = atr["model"]
scaler = atr["scaler"]
feature = atr["feature"]

# EDA - Обработка признаков
def clean_input_df(df):

    df_cl = df.copy()

    # Оставляем признаки, которые нужны для обучения модели
    df_cl = df_cl[feature]

    # Признаки, из которых необходимо извлечь число
    cleaning_features = ["mileage", "engine", "max_power"]

    # Применяем регулярные выражения для извлечения чисел
    for col in cleaning_features:
        if col in df_cl.columns:
            df_cl[col] = (df_cl[col].astype(str).str.extract(r"(\d+\.?\d*)")[0])

    # Приводим признаки к числовому формату
    num_cols = ["year", "km_driven", "seats"] + cleaning_features
    df_cl[num_cols] = df_cl[num_cols].apply(pd.to_numeric, errors="coerce") # Если есть типы данных, которые не преобразуются в число (например, str), то ставим NA

    # Заполняем пропуски в признаках медианами
    df_cl = df_cl.fillna(df_cl.median(numeric_only=True))

    return df_cl

# Приветственная "шапка" приложения
st.title("Привет! Это приложение для предсказания стоимости автомобиля")
st.markdown("Следуйте, пожалуйста, инструкциям, указанным ниже")

# Визуализация графиков
st.header("Визуализация графиков (EDA)")
uploaded_graphs = st.file_uploader("Загрузите CSV для визуализации данных", type=["csv"], key="graphs") # Блок для загрузки файла csv

if uploaded_graphs:
    df_graphs = pd.read_csv(uploaded_graphs)

    # Строим pairplot
    st.subheader("Попарные распределения числовых признаков")
    pairplot_atr = sns.pairplot(df_graphs[feature])
    st.pyplot(pairplot_atr)

    # Строим корреляционную матрицу (хитмап)
    st.subheader("Корреляционная матрица")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_graphs[feature].corr(), annot=True)
    st.pyplot(fig)

# Визуализация коэффициентов модели
st.header("Веса модели ElasticNet")

coef_df = pd.DataFrame({"Признак": feature, "Коэффициент": model.coef_}) # Формируем DataFrame Признак-коэффициент
st.write(coef_df)
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=coef_df, x="Коэффициент", y="Признак")
st.pyplot(fig)

# Предсказание цены автомобиля
st.header("Предсказание цены автомобиля")

# Блок выбора типа ввода данных
mode = st.radio("Выберите тип ввода данных:", ["Загрузить csv", "Ввести вручную"])

# Ветка загрузки csv файла
if mode == "Загрузить csv":
    uploaded = st.file_uploader("Загрузите CSV", type=["csv"], key="input_file") # Блок для загрузки файла csv

    if uploaded:
        df_upl = pd.read_csv(uploaded)
        df_clean = clean_input_df(df_upl)

        # Масштабируем входные данные
        X_scaled = scaler.transform(df_clean)
        
        # Предсказываем стоимость автомобиля
        preds = model.predict(X_scaled)

        st.success("Готово!")
        st.write(pd.DataFrame({"Предсказанная цена": preds}))

# Ветка ручного ввода данных
else:
    st.subheader("Введите данные:")

    #Пустой словарь записи значений ввода от пользователя
    input_values = {}
    for col in feature:
        input_values[col] = st.number_input(col) # Ключ - признак, базовое значение - 0

    # Превращаем словарь в DataFrame для дальнейшей работы
    df_inp = pd.DataFrame([input_values])

    X_scaled = scaler.transform(df_inp)
    pred = model.predict(X_scaled)[0]

    st.success(f"Предсказанная цена: {pred:,.0f}")