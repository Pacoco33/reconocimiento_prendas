import tensorflow as tf
import matplotlib.pyplot as plt

#Carga del dataset
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train, y_train), (X_test, y_test) = fashion_mnist

print(X_train.shape)
print(X_test.shape)

# Normalización
X_train, X_test = X_train / 255., X_test / 255.

#Visualización de una imagen
plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()

#Etiquetas de clases
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print("Etiqueta:", y_train[0], "-", class_names[y_train[0]])

#Mostrar múltiples imágenes
n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train[index]])
plt.show()

#Creación del modelo
tf.random.set_seed(42)

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=[28, 28]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=300, activation="relu"))
model.add(tf.keras.layers.Dense(units=100, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

#Compilación del modelo
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
)

#Entrenamiento del modelo
history = model.fit(X_train, y_train, epochs=30)

#Evaluación del modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Precisión en test:", test_acc)

#Predicción
#Tomar algunas imágenes del conjunto de test para predecir
X_new = X_test[:3]
y_proba = model.predict(X_new)
print("Probabilidades:\n", y_proba.round(2))
print("Predicciones:", tf.argmax(y_proba, axis=1).numpy())
print("Etiquetas reales:", y_test[:3])
