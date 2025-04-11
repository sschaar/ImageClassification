import numpy as np
import time
import PIL.Image as Image
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import datetime

from flower_classification import image_batch

# Schritt 1: Ein Bildklassifizierungsmodell von TensorFlow Hub laden
mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"

# Das MobileNetV2 Modell verwenden
classifier_model = mobilenet_v2

# Bildgröße definieren
IMAGE_SHAPE = (224, 224)

# Modell definieren, das auf TensorFlow Hub basiert
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])

# Schritt 2: Ein Bild herunterladen und auf das Modell anwenden
grace_hopper = tf.keras.utils.get_file('image.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)

# Bild anzeigen
grace_hopper

# Bild in ein NumPy-Array umwandeln und normalisieren
grace_hopper = np.array(grace_hopper)/255.0
grace_hopper.shape

# Schritt 3: Eine Batch-Dimension hinzufügen und das Bild durch das Modell schicken
result = classifier.predict(grace_hopper[np.newaxis, ...])
result.shape

# Schritt 4: Die Klasse mit der höchsten Wahrscheinlichkeit finden
predicted_class = tf.math.argmax(result[0], axis=-1)
predicted_class

# Schritt 5: Labels aus ImageNet decodieren
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

# Bild und Vorhersage anzeigen
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())

# Schritt 6: Einfaches Transfer Learning
# Jetzt erstellen wir eine benutzerdefinierte Klassifizierung für unser eigenes Dataset
# Blumen-Dataset herunterladen
import pathlib
data_file = tf.keras.utils.get_file(
  'flower_photos.tgz',
  'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
  cache_dir='.',
  extract=True)

data_root = pathlib.Path(data_file).with_suffix('')

# Bild-Dataset laden
batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# Die Klassenbezeichner ausgeben
class_names = np.array(train_ds.class_names)
print(class_names)

# Normalisierungsschicht erstellen
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Daten mit Prefetching und Caching optimieren
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Schritt 7: Modell vorbereiten
# Feature Extractor-Modell definieren (ohne die Klassifikationsschicht)
feature_extractor_model = mobilenet_v2

feature_extractor_layer = hub.KerasLayer(
    feature_extractor_model,
    input_shape=(224, 224, 3),
    trainable=False)  # Den Feature-Extractor einfrieren

# Batch von Bildern extrahieren
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

# Klassifikationskopf hinzufügen
num_classes = len(class_names)

model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(num_classes)
])

model.summary()

# Schritt 8: Das Modell kompilieren
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

# Schritt 9: Modell trainieren
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1)

NUM_EPOCHS = 10
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=NUM_EPOCHS,
                    callbacks=tensorboard_callback)

# Schritt 11: Modellvorhersagen auf einem Batch von Bildern machen
predicted_batch = model.predict(image_batch)
predicted_id = tf.math.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]
print(predicted_label_batch)

# Vorhersagen anzeigen
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_label_batch[n].title())
  plt.axis('off')
_ = plt.suptitle("Model predictions")

# Schritt 12: Modell exportieren und erneut laden
t = time.time()
export_path = "/tmp/saved_models/{}".format(int(t))
model.save(export_path)

# Modell erneut laden
reloaded = tf.keras.models.load_model(export_path)

# Vorhersagen nach dem Laden durchführen
reloaded_result_batch = reloaded.predict(image_batch)

# Überprüfen, ob die Vorhersagen nach dem Laden des Modells gleich sind
print(abs(reloaded_result_batch - predicted_batch).max())  # Vergleich der Vorhersageergebnisse

# Vorhersagen nach dem Laden anzeigen
reloaded_predicted_id = tf.math.argmax(reloaded_result_batch, axis=-1)
reloaded_predicted_label_batch = class_names[reloaded_predicted_id]
print(reloaded_predicted_label_batch)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(reloaded_predicted_label_batch[n].title())
  plt.axis('off')
_ = plt.suptitle("Model predictions")
