# ProyectoDL
Predicción del Éxito de Atracciones Turísticas: Un Enfoque de Deep Learning
# Proyecto: Predicción de Engagement en Puntos de Interés Turísticos / Project: Engagement Prediction for Tourist Points of Interest

## Descripción / Description

Este proyecto utiliza aprendizaje profundo para predecir el nivel de engagement de puntos de interés turísticos (POIs) combinando información visual de imágenes y datos contextuales (metadatos). La arquitectura del modelo integra dos redes convolucionales preentrenadas (EfficientNet y ResNet18) con una subred para metadatos, logrando una clasificación binaria entre alto y bajo engagement. / This project leverages deep learning to predict the engagement level of tourist points of interest (POIs) by combining visual information from images and contextual metadata. The model architecture integrates two pretrained convolutional neural networks (EfficientNet and ResNet18) with a dense subnetwork for metadata, achieving binary classification between high and low engagement.

---

## Características Principales / Key Features

- **Modelo Híbrido**: Combina EfficientNet y ResNet18 para procesar imágenes, con una subred densa para metadatos. / **Hybrid Model**: Combines EfficientNet and ResNet18 for image processing with a dense subnetwork for metadata.
- **Optimización Avanzada**: Uso de Optuna para ajustar hiperparámetros como la tasa de aprendizaje, dropout y peso de regularización. / **Advanced Optimization**: Utilizes Optuna for hyperparameter tuning, including learning rate, dropout, and regularization weight.
- **Regularización y Early Stopping**: Implementación de técnicas para evitar el sobreajuste. / **Regularization and Early Stopping**: Techniques to prevent overfitting.
- **Evaluación Completa**: Métricas como F1-score, precisión, recall y exactitud. / **Comprehensive Evaluation**: Metrics include F1-score, precision, recall, and accuracy.

---

## Requisitos / Requirements

- **Python 3.8+**
- Librerías principales: / Main libraries:
  - `torch`
  - `torchvision`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `Pillow`
  - `optuna`
  - `efficientnet_pytorch`

Instalar dependencias: / Install dependencies:

```bash
pip install -r requirements.txt
