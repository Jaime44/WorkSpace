import matplotlib.pyplot as plt



def plot_loss(history):
    """
    Función para dibujar la gráfica de pérdida en entrenamiento y validación.
    """
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida en entrenamiento y validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()


def contar_ocurrencias(df=None, col=None):
    """
    Función para contar las ocurrencias distintas en una columna de un DataFrame.

    Parámetros:
    - df: DataFrame de pandas.
    - col: Nombre de la columna en la que se contarán las ocurrencias.
    """
    count = 0
    # Verificar si la columna existe en el DataFrame
    if col is None or df is None:
        raise ValueError("Debes proporcionar un DataFrame y el nombre de la columna.")

    if col not in df.columns:
        raise ValueError(f"La columna '{col}' no existe en el DataFrame.")

    # Contar las ocurrencias distintas
    conteo_ocurrencias = df[col].value_counts()

    # Imprimir los valores únicos y su frecuencia
    for valor, frecuencia in conteo_ocurrencias.items():
        count += 1
        print(f"Valor: {valor}, Frecuencia: {frecuencia}")
    print(f'Total de valores distintos: {count}')

def mostrar_filas_por_valor(dataset, columna, valor):
    """
    Muestra las filas del conjunto de datos donde la columna tiene el valor dado.

    Parameters:
    - dataset: DataFrame, el conjunto de datos.
    - columna: str, el nombre de la columna a filtrar.
    - valor: valor que se desea para la columna.

    Returns:
    None
    """
    filas_filtradas = dataset[dataset[columna] == valor]
    print(filas_filtradas)

def eliminar_columnas(dataset, columnas_a_eliminar):
    """
    Elimina las columnas especificadas del conjunto de datos.

    Parameters:
    - dataset: DataFrame, el conjunto de datos.
    - columnas_a_eliminar: list, una lista de nombres de columnas a eliminar.

    Returns:
    DataFrame, el conjunto de datos con las columnas eliminadas.
    """
    dataset_sin_columnas = dataset.drop(columnas_a_eliminar, axis=1)
    return dataset_sin_columnas

def verificar_codificacion_tag(data_frame, columna_a_recorrer='tag', columna_condicional='tag', columna_deseada='tag_encoded'):
    """
    Verifica la codificación de la columna 'tag' en un DataFrame.

    Parameters:
    - data_frame: DataFrame, el conjunto de datos.
    - columna_a_recorrer: str, el nombre de la columna que se recorrerá (por defecto, 'tag').
    - columna_condicional: str, el nombre de la columna condicional (por defecto, 'tag').
    - columna_deseada: str, el nombre de la columna deseada (por defecto, 'tag_encoded').

    Returns:
    None
    """
    # Inicializar una lista para almacenar los valores únicos
    valores_distintos_tag = data_frame[columna_a_recorrer].unique()

    # Inicializar una lista para almacenar los resultados
    frecuencia_valores_distintos_tag_encode = []

    # Recorrer la lista de valores únicos
    for valor_unico in valores_distintos_tag:
        ocurrencias = len(data_frame[data_frame[columna_condicional] == valor_unico])

        # Encuentra la fila donde 'columna_condicional' tiene el valor 'valor_condicional'
        fila_deseada = data_frame.loc[data_frame[columna_condicional] == valor_unico]

        # Obtén el valor de 'columna_deseada' en esa fila
        valor_encode = fila_deseada[columna_deseada].values[0]

        # Contar ocurrencias de 'valor_encode'
        ocurrencias_encode = len(data_frame[data_frame[columna_deseada] == valor_encode])

        # Agregar el resultado a la lista de conteo de ocurrencias
        frecuencia_valores_distintos_tag_encode.append((valor_unico, valor_encode, ocurrencias_encode, ocurrencias))

    # Imprimir el resultado
    for valor_unico, valor_encode, ocurrencias_encode, ocurrencias in frecuencia_valores_distintos_tag_encode:
        if ocurrencias_encode != ocurrencias:
            print(f"{valor_unico}|{valor_encode} --> {ocurrencias_encode}=={ocurrencias} veces")
