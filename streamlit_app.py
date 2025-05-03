import streamlit as st
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Импортируем функции из основного модуля
from quantum_edge_detection import (
    resize_image,
    amplitude_encode,
    build_circuit,
    run_simulator,
    extract_edges,
)

# Интерфейс Streamlit
st.title("Квантовое детектирование границ")

# Загрузчик файла
uploader = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

# Вспомогательная функция визуализации через matplotlib с ближайшей интерполяцией
def show_image_matplotlib(img_array, cmap=None, title=None, figsize=(4, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)
    ax.axis('off')
    ax.imshow(img_array, cmap=cmap, interpolation='nearest')
    st.pyplot(fig)

# Инициализация состояния
if 'run_edge_detection' not in st.session_state:
    st.session_state.run_edge_detection = False

# Если файл загружен
if uploader is not None:
    # Показываем превью, если пока не запускали детектирование
    if not st.session_state.run_edge_detection:
        try:
            preview_img = Image.open(uploader)
            st.subheader("Предварительный просмотр")
            show_image_matplotlib(np.array(preview_img), title="Загруженное изображение")
        except Exception:
            st.warning("Не удалось отобразить превью изображения.")

    # Кнопка запуска детектирования
    if st.button("Запустить детектирование"):
        st.session_state.run_edge_detection = True

    # Когда кнопка нажата, выполняем детектирование и скрываем превью
    if st.session_state.run_edge_detection:
        # Сохраняем файл во временную директорию
        tfile = tempfile.NamedTemporaryFile(delete=False)
        uploader.seek(0)
        tfile.write(uploader.read())
        tfile.flush()

        # Обработка: вычисление карты границ
        rgb = resize_image(tfile.name).astype(np.uint8)
        gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
        data_qubits = int(np.log2(gray.size))
        coeffs_h = amplitude_encode(gray)
        coeffs_v = amplitude_encode(gray.T)
        qc_h = build_circuit(coeffs_h, data_qubits)
        qc_v = build_circuit(coeffs_v, data_qubits)
        with st.spinner("Запуск квантовой симуляции..."):
            statevecs = run_simulator([qc_h, qc_v])

        h_edges = extract_edges(statevecs[0], data_qubits)
        v_edges = extract_edges(statevecs[1], data_qubits).T
        edge_img = ((h_edges | v_edges) * 255).astype(np.uint8)

        # Отображаем только итоговые границы
        st.subheader("Итоговые границы")
        show_image_matplotlib(edge_img, cmap='gray')
        st.success("Обработка завершена.")
else:
    st.info("Пожалуйста, загрузите изображение для анализа.")