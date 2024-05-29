
import os
import numpy as np # type: ignore
from scipy.io import loadmat # type: ignore

# Librerias para trabajar con la red neuronal y procesamiento de datos
import tensorflow as tf # Para red neuronal profunda
import numpy as np
import matplotlib.pyplot as plt
import time # Para tomar el tiempo de entrenamiento de la red

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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




#####################################################################################
####################### Conversion a tensor tridimensional ##########################
#####################################################################################

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



#####################################################################################
#################### Verificacicon de "shape" - matrices pam y vsc ##################
#####################################################################################

# Directorios de salida
output_pam_dir_check = 'D:/TT/Memoria/waveletycnn/codigo_python/matrices_complejas_pam_procesadas'
output_vsc_dir_check = 'D:/TT/Memoria/waveletycnn/codigo_python/matrices_complejas_vsc_procesadas'

# Funcion para verificar la forma de una matriz
def verificar_shape(directorio, nombre_archivo):
    path = os.path.join(directorio, nombre_archivo)
    matriz = np.load(path)
    return matriz.shape

# Verificar la forma de un archivo de ejemplo en output_pam_dir_check
ejemplo_pam = os.listdir(output_pam_dir_check)[0]  # Obtener el primer archivo de la carpeta
shape_pam = verificar_shape(output_pam_dir_check, ejemplo_pam)
print(f"Shape de {ejemplo_pam} en {output_pam_dir_check}: {shape_pam}")

# Verificar la forma de un archivo de ejemplo en output_vsc_dir_check
ejemplo_vsc = os.listdir(output_vsc_dir_check)[0]  # Obtener el primer archivo de la carpeta
shape_vsc = verificar_shape(output_vsc_dir_check, ejemplo_vsc)
print(f"Shape de {ejemplo_vsc} en {output_vsc_dir_check}: {shape_vsc}")












####################################################################################
######################### Red Neuronal Profunda: U-net #############################
####################################################################################

# Directorios de entrada
input_pam_dir = 'D:/TT/Memoria/waveletycnn/codigo_python/matrices_complejas_pam_procesadas'
output_vsc_dir = 'D:/TT/Memoria/waveletycnn/codigo_python/matrices_complejas_vsc_procesadas'

# Función para cargar los archivos .npy
def load_npy_files(input_dir):
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')])
    data = [np.load(f) for f in files]
    return np.array(data)

# Cargar los datos de entrada y salida
X = load_npy_files(input_pam_dir) # inputs
Y = load_npy_files(output_vsc_dir) # outputs

# Verificar las formas de los datos cargados
print(f"Shape de los inputs (X): {X.shape}")
print(f"Shape de los outputs (Y): {Y.shape}")

# Definir la U-Net con regularización L2
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

# Definir la métrica NMSE ajustada para utilizar la varianza de los valores verdaderos
def nmse(y_true, y_pred):
    mse = tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred))
    var_true = tf.keras.backend.var(y_true)
    return mse / var_true

# Hiperparametros
max_epoch = 200
batchsize = 16
learning_rate = 0.001
validation_split = 0.3 # 70% entrenamiento & 30% validacion

# Definir el modelo
input_shape = X.shape[1:]  # forma del input a entrar. en este caso esta forma debe coincidir con las matrices que entran a la red
model = unet_model_with_l2(input_shape)

# DEFINICION DE LA FUNCION DE DECAIMIENTO, ALGORITMO OPTIMIZADOR, FUNCION DE PERDIDA Y METRICA
optimizer = tf.keras.optimizers.Adam(learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[nmse])

###########################################################################################################
###########################################################################################################
###########################################################################################################

# ENTRENAMIENTO DE LA RED
start_time = time.time()
history = model.fit(X, Y, epochs=max_epoch, batch_size=batchsize, validation_split=validation_split)
end_time = time.time()
total_time = end_time - start_time
print(f'Tiempo total de entrenamiento: {total_time:.2f} segundos')

###########################################################################################################
###########################################################################################################
###########################################################################################################

# Visualizar el NMSE
plt.plot(history.history['nmse'], label='NMSE (entrenamiento)')
plt.plot(history.history['val_nmse'], label='NMSE (validacion)')
plt.xlabel('Epoca')
plt.ylabel('NMSE')
plt.legend()
plt.show()
