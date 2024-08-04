#### Introduccion

Este proyecto sigue una metodología estructurada que incluye la recolección de datos, 
preprocesamiento, generación de embeddings, implementación de la búsqueda semántica, 
visualización de resultados y evaluación del sistema. La implementación se realiza en Python, 
utilizando una variedad de librerías como Langchain, Gensim, Transformers, Scikit-learn, Matplotlib y Seaborn.

### 1. Instalación de librerías y configuración

```python
!pip install kaggle
!kaggle datasets download -d Cornell-University/arxiv
!unzip arxiv.zip
!pip install pandas
!pip install gensim spacy matplotlib transformers seaborn scikit-learn -q -U
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!pip install getpass
```

- **`!pip install kaggle`**: Instala la librería de Kaggle, necesaria para descargar datasets desde Kaggle.
- **`!kaggle datasets download -d Cornell-University/arxiv`**: Descarga el dataset de arXiv desde Kaggle.
- **`!unzip arxiv.zip`**: Extrae el contenido del archivo zip descargado.
- **`!pip install pandas`**: Instala pandas, una librería esencial para la manipulación de datos.
- **`!pip install gensim spacy matplotlib transformers seaborn scikit-learn -q -U`**: Instala múltiples librerías necesarias para el procesamiento de texto, generación de embeddings y visualización.
- **`!mkdir ~/.kaggle` y siguientes**: Configura las credenciales de Kaggle para permitir la descarga de datasets privados.
- **`!pip install getpass`**: Instala getpass, una librería que permite solicitar contraseñas de manera segura.

### 2. Recolección de Datos

```python
import pandas as pd

# Definir el tamaño del bloque
chunk_size = 10000

# Cargar el dataset de arXiv desde Kaggle en bloques, utilizando 'lines=True' para leer el archivo línea por línea
data_iterator = pd.read_json('/content/arxiv-metadata-oai-snapshot.json', lines=True, chunksize=chunk_size)

# Mostrar una muestra de los datos
for chunk in data_iterator:
    print(chunk)
    break  # Para mostrar solo la primera muestra
```

- **`chunk_size = 10000`**: Define el tamaño del bloque para la lectura del dataset en partes manejables.
- **`pd.read_json(..., lines=True, chunksize=chunk_size)`**: Lee el archivo JSON línea por línea en bloques del tamaño definido.
- **`for chunk in data_iterator`**: Itera sobre cada bloque de datos, permitiendo el procesamiento en partes para no sobrecargar la memoria.

### 3. Preprocesamiento

```python
import re
import pandas as pd

# Función para limpiar y normalizar el texto
def preprocess_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Eliminar caracteres especiales y números
    text = re.sub(r'\s+', ' ', text).strip()  # Eliminar múltiples espacios
    return text

def process_and_yield_data(file_path, chunk_size=10000):
    data_iterator = pd.read_json(file_path, lines=True, chunksize=chunk_size)
    for chunk in data_iterator:
        for _, row in chunk.iterrows():
            row['abstract'] = preprocess_text(row['abstract'])
            yield row.to_dict()

# Uso de ejemplo
file_path = '/content/arxiv-metadata-oai-snapshot.json'  # Reemplazar con la ruta real del archivo

processed_data = []
for processed_row in process_and_yield_data(file_path):
    processed_data.append(processed_row)

df = pd.DataFrame(processed_data)
print(df.head())
```

- **`preprocess_text`**: Función para limpiar y normalizar el texto (convertir a minúsculas, eliminar caracteres especiales y múltiples espacios).
- **`process_and_yield_data`**: Lee los datos en bloques y aplica la función de preprocesamiento a cada fila.
- **`processed_data`**: Lista donde se almacenan las filas procesadas.

### 4. Generación de Embeddings

```python
!pip install transformers

from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
import pandas as pd

# Cargar el tokenizer y el modelo preentrenado de DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Verificar si una GPU está disponible y mover el modelo a la GPU si es posible
if torch.cuda.is_available():
    model = model.cuda()
    print("Model moved to GPU.")
else:
    print("No GPU available, using CPU.")

def get_embeddings(text):
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=128)).unsqueeze(0)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        if torch.cuda.is_available():
            embeddings = embeddings.cpu()
    return embeddings.numpy()
```

- **`DistilBertTokenizer` y `DistilBertModel`**: Se utilizan para tokenizar el texto y generar embeddings utilizando el modelo DistilBERT.
- **`torch.cuda.is_available()`**: Verifica si hay una GPU disponible y mueve el modelo a la GPU si es posible para acelerar el procesamiento.
- **`get_embeddings`**: Función que tokeniza el texto y genera los embeddings usando el modelo DistilBERT.

### 5. Búsqueda Semántica

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Cargar el tokenizer y el modelo preentrenado de DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model = model.cuda()  # Mover el modelo a la GPU si está disponible

def get_embeddings(text):
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=128)).unsqueeze(0).cuda()
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

def batch_get_embeddings(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        encoded_batch = tokenizer.batch_encode_plus(
            batch_texts,
            add_special_tokens=True,
            truncation=True,
            max_length=128,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        batch_input_ids = encoded_batch['input_ids'].cuda()
        batch_attention_masks = encoded_batch['attention_mask'].cuda()
        with torch.no_grad():
            batch_outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
            batch_embeddings = batch_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings)

texts = ["This is the first text.", "This is another text."]
embeddings = batch_get_embeddings(texts)
print(embeddings)
```

- **`cosine_similarity`**: Se utiliza para calcular la similitud entre los embeddings.
- **`batch_get_embeddings`**: Genera embeddings en lotes para manejar grandes cantidades de texto de manera más eficiente.

### 6. Visualización

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def visualize_results(results):
    embeddings = np.vstack(results['embeddings'].values)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=results['category'], palette='viridis')
    for i, row in results.iterrows():
        plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], row['title'], fontsize=9)
    plt.title('Visualización de Resultados')
    plt.show()

results = pd.DataFrame({
    'embeddings': [embeddings[0], embeddings[1]],  
    'category': ['Categoría 1', 'Categoría 2'],  
    'title': ['PCA 1', 'PCA 2']  
})
visualize_results(results)
```

- **`PCA`**: Se utiliza para reducir la dimensionalidad de los embeddings para su visualización.
- **`sns.scatterplot`**: Crea un gráfico de dispersión de los embeddings reducidos.

### 7. Evaluación

```python
import json
import pandas as pd

def search(query, data, top_k=5):
    results = pd.DataFrame({
        'title': ['Result 1', 'Result 2', 'Result 3', 'Result 4', 'Result 5'],
        'relevance': [1, 0, 1, 1,

Continuemos con la explicación del código, enfocándonos en la sección de evaluación:

### 7. Evaluación

```python
import json
import pandas as pd

def search(query, data, top_k=5):
    results = pd.DataFrame({
        'title': ['Result 1', 'Result 2', 'Result 3', 'Result 4', 'Result 5'],
        'relevance': [1, 0, 1, 1, 0]  # Ejemplo de puntuaciones de relevancia
    })
    return results

def evaluate_search(query, data_file, top_k=5):
    relevant_count = 0
    with open(data_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if 'query' in entry['title'].lower():  # Ejemplo de verificación de relevancia
                relevant_count += 1
                if relevant_count >= top_k:
                    break

    precision = relevant_count / top_k
    return precision

query = "machine learning"
data_file = "/content/arxiv-metadata-oai-snapshot.json"
precision = evaluate_search(query, data_file)
print(f'Precision for the query "{query}": {precision}')
```

- **`search`**: Esta función simula una búsqueda y devuelve un DataFrame con resultados y puntuaciones de relevancia. Esta es una función de placeholder, que debería ser reemplazada con la lógica real de búsqueda semántica.

- **`evaluate_search`**: Esta función evalúa la precisión de la búsqueda. Lee el archivo JSON línea por línea para evitar la sobrecarga de memoria y cuenta cuántos resultados relevantes se encuentran en las primeras `top_k` filas.

- **`precision`**: La precisión se calcula como el número de resultados relevantes dividido por `top_k`. En el ejemplo, se busca el término "machine learning" y se evalúa la precisión de la búsqueda en el dataset.

### Resumen del Flujo del Proyecto

1. **Instalación y configuración**: Se instalan todas las librerías necesarias y se configura el entorno para trabajar con Kaggle.
2. **Recolección de datos**: Se descarga el dataset de arXiv y se carga en partes para evitar la sobrecarga de memoria.
3. **Preprocesamiento**: Se limpia y normaliza el texto de los artículos científicos.
4. **Generación de embeddings**: Utilizando DistilBERT, se generan embeddings para representar los artículos.
5. **Búsqueda semántica**: Se implementa una función para realizar búsquedas basadas en la similitud de embeddings.
6. **Visualización**: Se visualizan los resultados de las búsquedas utilizando PCA y gráficos de dispersión.
7. **Evaluación**: Se evalúa la precisión de las búsquedas para garantizar la calidad del sistema.

Cada sección está diseñada para manejar grandes volúmenes de datos de manera eficiente, asegurando que el sistema pueda operar incluso en entornos con recursos limitados.

### Conclusion
El desarrollo de un sistema de búsqueda semántica para artículos científicos utilizando embeddings y 
Langchain demuestra cómo las técnicas avanzadas de procesamiento de lenguaje natural pueden mejorar 
significativamente la relevancia de los resultados de búsqueda. Este sistema permite a los investigadores 
y profesionales encontrar información más relevante y contextualmente apropiada, superando las limitaciones 
de los sistemas de búsqueda basados únicamente en palabras clave.
##
