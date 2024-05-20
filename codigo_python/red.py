
import os
import numpy as np # type: ignore
from scipy.io import loadmat # type: ignore

# Librerias para trabajar con la red neuronal y procesamiento de datos
import tensorflow as tf # Para red neuronal profunda
import numpy as np
import matplotlib.pyplot as plt

####################################################################################
####################### Convertir matrices de .mat a .npy ##########################
####################################################################################

# Directorios de entrada y salida
input_pam_dir = 'D:/TT/Memoria/waveletycnn/codigo_matlab/codigo_fuente/matrices_complejas_pam_mat' # matrices en formato .mat
input_vsc_dir = 'D:/TT/Memoria/waveletycnn/codigo_matlab/codigo_fuente/matrices_complejas_vsc_mat' # matrices en formato .mat
output_pam_dir = 'D:/TT/Memoria/waveletycnn/codigo_python/matrices_complejas_pam_python' # INPUT PARA LA RED
output_vsc_dir = 'D:/TT/Memoria/waveletycnn/codigo_python/matrices_complejas_vsc_python' # OUTPUT O SALIDAS ESPERADAS PARA LA RED

# Crear los directorios de salida si no existen
os.makedirs(output_pam_dir, exist_ok=True)
os.makedirs(output_vsc_dir, exist_ok=True)

# Funcion para convertir archivos .mat a .npy
def convert_mat_to_npy(input_dir, output_dir, prefix):
    for i in range(1, 31):
        mat_file = os.path.join(input_dir, f'{prefix}_noise_{i}.mat')
        npy_file = os.path.join(output_dir, f'{prefix}_noise_{i}.npy')
        
        # Cargar el archivo .mat
        mat_data = loadmat(mat_file)
        
        # Extraer la matriz compleja
        matrix_key = [key for key in mat_data.keys() if not key.startswith('__')][0]
        matrix = mat_data[matrix_key]
        
        # Guardar la matriz en formato .npy
        np.save(npy_file, matrix)

# Convertir archivos .mat a .npy para PAM y VSC
convert_mat_to_npy(input_pam_dir, output_pam_dir, 'matrix_complex_pam')
convert_mat_to_npy(input_vsc_dir, output_vsc_dir, 'matrix_complex_vsc')




####################################################################################
####################### Conversion a tensor tridimensional##########################
####################################################################################

# Directorios de entrada y salida
input_pam_dir = 'D:/TT/Memoria/waveletycnn/codigo_python/matrices_complejas_pam_python'
input_vsc_dir = 'D:/TT/Memoria/waveletycnn/codigo_python/matrices_complejas_vsc_python'
output_pam_dir = 'D:/TT/Memoria/waveletycnn/codigo_python/matrices_complejas_pam_procesadas'
output_vsc_dir = 'D:/TT/Memoria/waveletycnn/codigo_python/matrices_complejas_vsc_procesadas'

# Crear directorios de salida si no existen
os.makedirs(output_pam_dir, exist_ok=True)
os.makedirs(output_vsc_dir, exist_ok=True)

# Función para procesar las matrices complejas
def procesar_matriz_compleja(matriz_compleja):
    datos_organizados = np.stack((matriz_compleja.real, matriz_compleja.imag), axis=-1)
    return datos_organizados

# Procesar matrices complejas en la carpeta input_pam_dir
for filename in os.listdir(input_pam_dir):
    if filename.endswith('.npy'):
        input_path = os.path.join(input_pam_dir, filename)
        output_path = os.path.join(output_pam_dir, filename)
        
        # Cargar la matriz compleja
        matriz_compleja = np.load(input_path)
        
        # Procesar la matriz compleja
        datos_organizados = procesar_matriz_compleja(matriz_compleja)
        
        # Guardar los datos procesados
        np.save(output_path, datos_organizados)

# Procesar matrices complejas en la carpeta input_vsc_dir
for filename in os.listdir(input_vsc_dir):
    if filename.endswith('.npy'):
        input_path = os.path.join(input_vsc_dir, filename)
        output_path = os.path.join(output_vsc_dir, filename)
        
        # Cargar la matriz compleja
        matriz_compleja = np.load(input_path)
        
        # Procesar la matriz compleja
        datos_organizados = procesar_matriz_compleja(matriz_compleja)
        
        # Guardar los datos procesados
        np.save(output_path, datos_organizados)

print("Procesamiento completado.")





####################################################################################
######################### Red Neuronal Profunda: U-net #############################
####################################################################################

# Directorios de entrada
input_pam_dir = 'D:/TT/Memoria/waveletycnn/codigo_python/matrices_complejas_pam_python'
output_vsc_dir = 'D:/TT/Memoria/waveletycnn/codigo_python/matrices_complejas_vsc_python'

# Cargar matrices .npy
def load_npy_files(input_dir):
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')])
    data = [np.load(f) for f in files]
    return np.array(data)

# Cargar los datos de entrada y salida
X = load_npy_files(input_pam_dir)
Y = load_npy_files(output_vsc_dir)

# Dividir las matrices en partes reales e imaginarias
X_real_imag = np.stack((X.real, X.imag), axis=-1)
Y_real_imag = np.stack((Y.real, Y.imag), axis=-1)

def unet_model_with_l2(input_shape, l2_lambda):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Regularizer
    l2_reg = tf.keras.regularizers.l2(l2_lambda)
    
    # Encoder
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(inputs)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(p1)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(p2)
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(c3)
    
    # Decoder
    u4 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = tf.keras.layers.concatenate([u4, c2])
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(u4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(c4)
    
    u5 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = tf.keras.layers.concatenate([u5, c1])
    c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(u5)
    c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(c5)
    
    outputs = tf.keras.layers.Conv2D(2, (1, 1), activation='linear')(c5)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model

def nmse(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred)) / tf.keras.backend.mean(tf.keras.backend.square(y_true))

# Hiperparámetros
max_epoch = 100
size_batch = 8
learning_rate = 0.001
l2_lambda = 0.01
validation_split = 0.1

# Definir el modelo
input_shape = X_real_imag.shape[1:]  # Asegúrate de que esto coincide con la forma de tus datos
model = unet_model_with_l2(input_shape, l2_lambda)

# Configurar el optimizador con la tasa de aprendizaje especificada
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compilar el modelo con la métrica NMSE
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[nmse])

# Entrenar el modelo
history = model.fit(X_real_imag, Y_real_imag, epochs=max_epoch, batch_size=size_batch, validation_split=validation_split)

# Visualizar el NMSE
plt.plot(history.history['nmse'], label='NMSE (entrenamiento)')
plt.plot(history.history['val_nmse'], label='NMSE (validación)')
plt.xlabel('Época')
plt.ylabel('NMSE')
plt.legend()
plt.show()
