#!/usr/bin/env python3
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveStatevector


def resize_image(input_path: str, size: tuple = (32, 32)) -> np.ndarray:
    """
    Загружает RGB-изображение, изменяет размер до заданного и возвращает массив (H, W, 3).
    Убеждается, что размер соответствует указанному.
    """
    img = Image.open(input_path).convert('RGB')
    resized = img.resize(size)
    # Проверяем, что новое изображение имеет корректные размеры
    assert resized.size == size, f"Resized image size {resized.size} != expected {size}"
    return np.asarray(resized)


def plot_image(image_array: np.ndarray, title: str) -> None:
    # Визуализирует 2D или 3D массив как изображение с заголовком.
    plt.figure()
    plt.title(title)
    plt.axis('off')
    plt.imshow(image_array, cmap=None if image_array.ndim == 3 else 'gray')
    plt.show()


def amplitude_encode(img_data: np.ndarray) -> np.ndarray:
    # Нормализует данные для амплитудной кодировки.
    flat = img_data.flatten().astype(np.float64)
    norm = np.linalg.norm(flat)
    return flat / norm if norm > 0 else flat


def build_circuit(coeffs: np.ndarray, data_qubits: int, ancilla_qubits: int = 1) -> QuantumCircuit:
    # Строит квантовую схему с инициализацией и операциями для сканирования.
    total_qubits = data_qubits + ancilla_qubits
    qc = QuantumCircuit(total_qubits)
    qc.initialize(coeffs, list(range(ancilla_qubits, total_qubits)))
    qc.h(0)
    perm = np.eye(2**total_qubits)
    perm = np.roll(perm, 1, axis=1)
    qc.unitary(perm, range(total_qubits))
    qc.h(0)
    qc.append(SaveStatevector(qc.num_qubits), qc.qubits)
    return qc


def run_simulator(circuits: list) -> list:
    # Запускает схемы на statevector-симуляторе и возвращает векторы состояний.
    sim = AerSimulator(method='statevector')
    transpiled = transpile(circuits, sim)
    job = sim.run(transpiled)
    result = job.result()
    return [result.get_statevector(i) for i in range(len(circuits))]


def extract_edges(statevec: np.ndarray, data_qubits: int) -> np.ndarray:
    """
    Формирует двоичное изображение границ из вектора состояния.
    Использует вероятность выхода анциллярного кубита в состояние |1>.
    """
    # Вычисляем вероятности всех состояний
    probs = np.abs(statevec)**2
    # Берём вероятности, где младший бит (анциллярный) равен 1
    edge_probs = probs[1::2]
    # Определяем порог как среднюю вероятность границы
    thresh = edge_probs.mean()
    # Генерируем бинарное изображение границ
    edges = (edge_probs > thresh).astype(int)
    # Восстанавливаем квадратную форму изображения
    size = int(np.sqrt(edges.size))
    return edges.reshape(size, size)

def main():
    # Путь к входному изображению: по умолчанию Apple3.jpg или передается аргументом
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'img.png'
    size = (32, 32)

    # Шаг 1: ресайзинг RGB-изображения
    rgb_img = resize_image(input_path, size)
    plot_image(rgb_img, 'Resized RGB Image')

    # Шаг 2: перевод в оттенки серого для квантового сканирования
    gray = np.dot(rgb_img[..., :3], [0.2989, 0.5870, 0.1140])
    plot_image(gray, 'Grayscale Image')

    # Подготовка к квантовым схемам
    data_qubits = int(np.log2(gray.size))
    coeffs_h = amplitude_encode(gray)
    coeffs_v = amplitude_encode(gray.T)

    qc_h = build_circuit(coeffs_h, data_qubits)
    qc_v = build_circuit(coeffs_v, data_qubits)

    # Запуск симуляции
    statevecs = run_simulator([qc_h, qc_v])

    # Извлечение границ для горизонтального сканирования
    h_edges = extract_edges(statevecs[0], data_qubits)
    # Визуализация горизонтального скана
    plot_image(h_edges, 'Horizontal scan output')

    # Извлечение границ для вертикального сканирования
    v_edges = extract_edges(statevecs[1], data_qubits).T
    # Визуализация вертикального скана
    plot_image(v_edges, 'Vertical scan output')

    # Объединение результатов сканирования
    edge_img = h_edges | v_edges
    # Визуализация итогового изображения границ
    plot_image(edge_img, 'Edge-Detected image')

if __name__ == '__main__':
    main()