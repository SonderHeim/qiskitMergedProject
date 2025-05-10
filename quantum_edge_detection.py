#!/usr/bin/env python3
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveStatevector
from qiskit.circuit.library import QFT
from skimage.filters import threshold_otsu


def resize_image(input_path: str, size: tuple = (256, 256)) -> np.ndarray:
    img = Image.open(input_path).convert('RGB')
    resized = img.resize(size)
    assert resized.size == size, f"Resized image size {resized.size} != expected {size}"
    return np.asarray(resized)


def plot_image(image_array: np.ndarray, title: str) -> None:
    plt.figure()
    plt.title(title)
    plt.axis('off')
    cmap = None if image_array.ndim == 3 else 'gray'
    plt.imshow(image_array, cmap=cmap)
    plt.show()


def binarize_image(gray: np.ndarray) -> np.ndarray:
    thresh = threshold_otsu(gray)
    return (gray > thresh).astype(float)


def amplitude_encode(img_data: np.ndarray) -> np.ndarray:
    flat = img_data.flatten().astype(np.float64)
    norm = np.linalg.norm(flat)
    return flat / norm if norm > 0 else flat


def build_circuit(coeffs: np.ndarray, data_qubits: int, ancilla_qubits: int = 1) -> QuantumCircuit:
    # Циклический сдвиг через контролируемый аддер (QFT)
    total_qubits = data_qubits + ancilla_qubits
    qc = QuantumCircuit(total_qubits)
    # Инициализация данных на дата-кубитах
    data_idx = list(range(ancilla_qubits, total_qubits))
    qc.initialize(coeffs, data_idx)
    # Анцилла
    anc = 0
    qc.h(anc)
    # QFT на регистре данных
    qc.append(QFT(num_qubits=data_qubits, do_swaps=False).to_gate(label='QFT'), data_idx)
    # Контролируемый аддер: при ancilla=1 добавить 1 к регистру
    for i, dq in enumerate(data_idx):
        angle = 2 * np.pi / (2 ** (i + 1))
        qc.cp(angle, anc, dq)
    # Обратный QFT
    qc.append(QFT(num_qubits=data_qubits, do_swaps=False).inverse().to_gate(label='iQFT'), data_idx)
    qc.h(anc)
    # Сохранение стейтовектора
    qc.append(SaveStatevector(qc.num_qubits), qc.qubits)
    return qc


def run_simulator(circuits: list) -> list:
    sim = AerSimulator(method='statevector')
    transpiled = transpile(circuits, sim)
    job = sim.run(transpiled)
    result = job.result()
    return [result.get_statevector(i) for i in range(len(circuits))]


def extract_edges(statevec: np.ndarray, data_qubits: int) -> np.ndarray:
    probs = np.abs(statevec)**2
    edge_probs = probs[1::2]
    thresh = edge_probs.mean()
    edges = (edge_probs > thresh).astype(int)
    size = int(np.sqrt(edges.size))
    return edges.reshape(size, size)


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'carBlack.png'
    size = (256, 256)

    # Шаг 1: загрузка и ресайзинг
    rgb_img = resize_image(input_path, size)
    plot_image(rgb_img, 'Resized RGB Image')

    # Шаг 2: в оттенки серого
    gray = np.dot(rgb_img[..., :3], [0.2989, 0.5870, 0.1140])
    plot_image(gray, 'Grayscale Image')

    # Предобработка: бинаризация
    binary = binarize_image(gray)
    plot_image(binary, 'Binarized Image')

    # Квантовая часть на бинаризованном изображении
    data_qubits = int(np.log2(binary.size))
    coeffs_h = amplitude_encode(binary)
    coeffs_v = amplitude_encode(binary.T)

    qc_h = build_circuit(coeffs_h, data_qubits)
    qc_v = build_circuit(coeffs_v, data_qubits)

    statevecs = run_simulator([qc_h, qc_v])

    # Извлечение краёв
    h_edges = extract_edges(statevecs[0], data_qubits)
    plot_image(h_edges, 'Horizontal scan output')

    v_edges = extract_edges(statevecs[1], data_qubits).T
    plot_image(v_edges, 'Vertical scan output')

    edge_img = h_edges | v_edges
    plot_image(edge_img, 'Edge-Detected image')

if __name__ == '__main__':
    main()