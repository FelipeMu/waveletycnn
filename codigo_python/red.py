import os
import numpy as np # type: ignore
from scipy.io import loadmat # type: ignore

# Directorios de entrada y salida
input_pam_dir = 'D:/TT/Memoria/waveletycnn/codigo_matlab/codigo_fuente/matrices_complejas_pam_mat'
input_vsc_dir = 'D:/TT/Memoria/waveletycnn/codigo_matlab/codigo_fuente/matrices_complejas_vsc_mat'
output_pam_dir = 'D:/TT/Memoria/waveletycnn/codigo_python/matrices_complejas_pam_python'
output_vsc_dir = 'D:/TT/Memoria/waveletycnn/codigo_python/matrices_complejas_vsc_python'

# Crear los directorios de salida si no existen
os.makedirs(output_pam_dir, exist_ok=True)
os.makedirs(output_vsc_dir, exist_ok=True)

# Función para convertir archivos .mat a .npy
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
