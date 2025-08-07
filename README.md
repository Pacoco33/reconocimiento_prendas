# Reconocimiento de Prendas con TensorFlow

Este proyecto implementa una red neuronal para el reconocimiento de imágenes de prendas de ropa utilizando el dataset **Fashion MNIST**.

## Descripción

Se trata de un modelo de clasificación multicategoría que recibe imágenes de 28x28 píxeles y predice a qué clase de prenda pertenece cada una. El proyecto incluye todo el flujo de trabajo: carga y visualización de datos, creación del modelo, entrenamiento, evaluación y predicción.

## Dataset

- **Fashion MNIST**: conjunto de datos proporcionado por `tensorflow.keras.datasets`
- Contiene imágenes en escala de grises de 10 clases de ropa distintas:
  - T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Arquitectura del modelo

- Entrada: 28x28 píxeles
- Capa de flatten (aplana la imagen)
- Capa densa: 300 neuronas, activación ReLU
- Capa densa: 100 neuronas, activación ReLU
- Capa de salida: 10 neuronas, activación Softmax

## Entrenamiento

- Pérdida: `sparse_categorical_crossentropy`
- Optimizador: `sgd`
- Métrica: `accuracy`
- Épocas: 30

## Resultados

Tras el entrenamiento, el modelo alcanza una **precisión de test** cercana al 87–89% dependiendo de la ejecución.

## Visualización

Se muestran imágenes individuales y una cuadrícula de imágenes con sus etiquetas correspondientes para entender mejor el conjunto de entrenamiento.

## Requisitos

- Python
- TensorFlow
- Matplotlib

Instalación recomendada:
```bash
pip install tensorflow matplotlib
```

