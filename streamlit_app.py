import streamlit as st
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Классификатор
from infer_integration import load_model, predict_image

# Квантовые функции
from quantum_edge_detection import (
    resize_image,
    amplitude_encode,
    build_circuit,
    run_simulator,
    extract_edges,
)

st.title("Квантовое обнаружение границ + классификация")

# Загрузчик файла
uploader = st.file_uploader("Загрузите изображение(я)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Вспомогательная функция визуализации через matplotlib с ближайшей интерполяцией
def show_image_matplotlib(img_array, cmap=None, title=None, figsize=(4, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)
    ax.axis('off')
    ax.imshow(img_array, cmap=cmap, interpolation='nearest')
    st.pyplot(fig)

if uploader:
    files = uploader if isinstance(uploader, list) else [uploader]
    st.subheader("Предварительный просмотр")
    for f in files:
        img = Image.open(f)
        show_image_matplotlib(np.array(img), title=f.name)

    if st.button("Запустить детектирование для всех"):
        model, device = load_model()
        for f in files:
            st.markdown(f"---\n## Обработка {f.name}")
            # 1) Сохраняем загруженный файл во временный файл
            tfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            f.seek(0)
            tfile.write(f.read())
            tfile.flush()

            # 2) Квантовое обнаружение границ
            rgb = resize_image(tfile.name).astype(np.uint8)
            gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
            data_qubits = int(np.log2(gray.size))
            coeffs_h = amplitude_encode(gray)
            coeffs_v = amplitude_encode(gray.T)
            qc_h = build_circuit(coeffs_h, data_qubits)
            qc_v = build_circuit(coeffs_v, data_qubits)

            progress = st.progress(0)
            statevecs = []
            for idx, qc in enumerate([qc_h, qc_v], 1):
                with st.spinner(f"Симуляция {idx}/2 для {f.name}..."):
                    res = run_simulator([qc])
                    statevecs.append(res[0])
                progress.progress(idx * 50)
            progress.empty()

            h_edges = extract_edges(statevecs[0], data_qubits)
            v_edges = extract_edges(statevecs[1], data_qubits).T
            edge_img = ((h_edges | v_edges) * 255).astype(np.uint8)

            st.subheader(f"Результат квантового обнаружения границ: {f.name}")
            show_image_matplotlib(edge_img, cmap='gray')

            # 3) Классификация исходного изображения
            label = predict_image(model, device, Image.open(f))
            st.subheader(f"Классификация: {f.name}")
            st.markdown(f"**{label.upper()}**")

        st.success("Обработка всех изображений завершена.")
else:
    st.info("Пожалуйста, загрузите одно или несколько изображений.")