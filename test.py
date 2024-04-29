import csv
import os
import pywt # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
# plot fourier transform for comparison
from numpy.fft import rfft, rfftfreq # type: ignore
# Para pasar plot a imagen rgb
from PIL import Image # type: ignore
from io import BytesIO

# Para el trabajo con archivos
import pandas as pd # type: ignore
# Para la aplicacion de ruido gaussiano y filtros
from scipy.signal import butter, filtfilt # type: ignore
from random import uniform


##############################################
################# FUNCIONES ##################
##############################################

# Se procede a realizar preprocesamiento de los datos a estudiar
'''
Nombre de funcion: is_power_of_two()
Descripcion: funcion encargada de verificar si la cantidad de muestras de una senal es potencia de 2
Entrada: 
    number [int]: cantidad de muestras de la senal
    signal_name [string]: nombre de la senal que se esta analizando, en este caso la senal de velocidad sanguinea cerebral
Salida: void
'''
def is_power_of_two(number, signal_name):
  # Verifica si el logaritmo en base 2 del numero es un numero entero
  if np.log2(number) % 1 == 0:
      print('El largo de', signal_name, 'es potencia de 2')
  else:
      print('El largo de', signal_name, 'no es potencia de 2. Por favor corregir la cantidad de muestras de la senal.')



'''
Nombre de funcion: apply_fft()
Descripcion: funcion encargada de aplicar la fft y mostrar grafico amplitud vs freq
Entrada:
   signal [array of float]: senal vsc a analizar. Contiene las muestras o puntos de la senal
   sampling_period [float]: periodo de muestreo. Cada cuanto tiempo de guardo una muestra de la senal
 Salida:
   - plot: grafico amplitud vs freq de la senal de entrada
'''
def apply_fft(signal, ts):
  # Paso 1: Aplicacion de la Transformada de Fourier
  transformada = np.fft.fft(signal)
  # Paso 2: Calcular las frecuencias correspondientes
  frecuencias = np.fft.fftfreq(len(signal), d=ts)

  freq_max = max(frecuencias)
  freq_min = 2.50
  abs_freq = np.abs(frecuencias)
  for f in abs_freq:
    if f <= freq_min and f != 0.00:
      freq_min = f
  print('La frecuencia minima es:',freq_min)
  print('La frecuencia maxima es:',freq_max)

  # Paso 5: Graficar amplitud vs. frecuencia
  plt.plot(frecuencias, np.abs(transformada))
  plt.xlabel('Frecuencia (Hz)')
  plt.ylabel('Amplitud')
  plt.title('Transformada de Fourier')
  plt.grid(True)
  # Mostrando la frecuencia maxima de la senal
  plt.text(0.95, 0.95, f'freq_max: {freq_max:.2f} Hz', transform=plt.gca().transAxes,
          fontsize=10, ha='right', va='top')
  plt.show()





'''
Nombre de funcion: apply_noise_and_filter()
Descripcion: funcion encargada de generar ruido y aplicar el filtro Butterworth
Entrada:
  signal[array of float]: senal vsc a la que se le aplicara ruido y filtro
Salida:
  filteres_signal[array of float]: senal vsc luego de aplicar ruido y filtro
'''
def apply_noise_and_filter(signal):
    # hiper-parametros para el proceso de aplicacion de ruido y filtro buterworth
    coef_range=(0.05, 0.10)
    order=8
    cutoff=0.25
    fs=5.0
    # Generar un coeficiente de variación aleatorio
    coef = uniform(*coef_range)

    # Calcular el ruido aleatorio
    noise = np.random.normal(0, coef * np.std(signal), len(signal))

    # Agregar el ruido a la señal
    noisy_signal = signal + noise

    # Crear un filtro Butterworth
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyquist  # Normalizar la frecuencia de corte
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Aplicar el filtro pasabajos de fase cero
    filtered_signal = filtfilt(b, a, noisy_signal)

    return filtered_signal






######################################################
############## PRE-PROCESAMIENTO #####################
######################################################


# periodo de muestreo
ts = 0.2 # seg

# frecuencia de muestreo
fs = 1.0/ts # Hz

print('Periodo de muestreo de señales:', ts, '[seg]')
print('Frecuencia de muestreo de señales:', fs, '[Hz]\n')
# Arreglo para almcenar los datos de las senales
pam = [] # PAM: Presión Arterial Media
vsc = [] # VSC: Velocidad Sanguínea Cerebral

#=================================================

folder_csv = 'D:/TT/Memoria/waveletycnn/signals'
files_csv = os.listdir(folder_csv)


# Visualizar los archivos existentes
print('Archivos encontrados:\n')
print(files_csv)


# Se procede a almacenar cada archivo .csv en un diccionario y luego en un arreglo de senales
signals = []
for each_csv in files_csv:
    dicc_signal_aux = {}

    carpeta_csv = 'D:/TT/Memoria/waveletycnn/signals'
    archivo_csv = each_csv
    #concatenar ruta y csv para acceder a los datos
    ruta_archivo = os.path.join(carpeta_csv, archivo_csv)
    # lectura de los archivos


    # Arreglo para almcenar los datos de las senales
    pam = [] # PAM: Presión Arterial Media
    vsc = [] # VSC: Velocidad Sanguínea Cerebral

    # Abre el archivo CSV en modo lectura
    with open(ruta_archivo, newline='') as csvfile:
        # Lee el archivo CSV usando el lector CSV
        csv_reader = csv.reader(csvfile)

        # Itera sobre cada fila en el archivo CSV
        for row in csv_reader:
            # Cada fila se convierte en una lista de valores, donde cada valor representa una celda en esa fila
            pam.append(row[0])
            vsc.append(row[1])

    #Se elimina los nombre de las filas PAM y VFSC que estaban contenidas dentro de los arreglos
    del pam[0]
    del vsc[0]

    pam = np.array(pam)
    pam = pam.astype(np.float64)
    # Se toma un largo de puntos de potencia de 2, de esta manera existe una
    # compatibilidad con el nivel de descomposión otorgado a la T.Wavelet
    pam = pam[0:1024]

    # Se toma un largo de puntos de potencia de 2, de esta manera existe una
    # compatibilidad con el nivel de descomposión otorgado a la T.Wavelet
    vsc = np.array(vsc)
    vsc = vsc.astype(np.float64)
    vsc = vsc[0:1024]

    # se guarda el nombre del archivo y la senal de pam en el diccioanrio
    dicc_signal_aux['name_file'] = each_csv
    dicc_signal_aux['signal_pam'] = pam
    dicc_signal_aux['signal_vsc'] = vsc

    signals.append(dicc_signal_aux)


print('\n**** mostrando el arreglo de diccionarios de csv****\n')

# verificnado contenidos del arreglo de diccionarios de csv's
for dicc in signals:
  print('nombre del archivo: ',dicc['name_file'])
  print('señal PAM: ', dicc['signal_pam'], '- N° de instancias', len(dicc['signal_pam']))
  print('señal vsc: ', dicc['signal_vsc'], '- N° de instancias', len(dicc['signal_vsc']))
  print('----------------------------------------')
print('----------------------------------------')
print('----------------------------------------')

##################################################################################################
##################################################################################################
##################################################################################################


# Se analizan si tienen el mismo largo ambos arreglos (pam y vsc) y que sean "potencias de 2"
for dicc in signals:
  len_distinto = False
  if len(dicc['signal_pam']) != len(dicc['signal_vsc']):
    len_distinto = True
    print('Arreglos PAM y VSC con distinto largo.\n')
  print(dicc['name_file'], '|| Numero de muestras:', len(dicc['signal_pam']))
  is_power_of_two(len(dicc['signal_pam']), 'señal PAM')
  is_power_of_two(len(dicc['signal_vsc']), 'señal VSC')
  print('\n')

##################################################################################################
##################################################################################################
##################################################################################################

# IDENTIFICACION DE LAS WAVELET MADRES A UTILIZAR EN LA APLICACIÓN DE CWT (Continuous Wavelet Transform)
wavelist = pywt.wavelist(kind='continuous')
print(wavelist)

##################################################################################################
##################################################################################################
##################################################################################################

# Aplicacion de CWT y espectros
for s in signals:
  print('\n')
  print('SEÑAL:', s['name_file'])
  # Paso 1: Preparar las entradas para la aplicacion de la fft y CWT
  n = len(s['signal_vsc'])  # cantidad de muestras de la senal
  t = np.linspace(0, n * ts, n)  # tiempo, con un periodo de muestreo de 0.2 segundos
  signal_to_analize = s['signal_vsc']


  # Paso 2: analizar FFT
  # Llamada a funcion que aplica la transformada de Fourier
  apply_fft(signal_to_analize, ts)

  # Paso 3: Elegir la wavelet y las escalas para la CWT
  wavelet = 'morl'  # tipo de wavelet
  scales = np.arange(1, 513)  # rango de escalas a aplicar

  # Paso 4: Aplicar la CWT
  coeficientes, frecuencias = pywt.cwt(signal_to_analize, scales, wavelet, sampling_period=ts)


  # Paso 5: Graficar el resultado de la CWT como un espectrograma
  plt.figure(figsize=(12, 6))
  plt.imshow(np.abs(coeficientes), aspect='auto', extent=[0, t[-1], scales[0], scales[-1]], cmap='jet')
  plt.colorbar(label='Magnitud')
  plt.xlabel('Tiempo (s)')
  plt.ylabel('Escala')
  plt.title('Transformada Wavelet Continua (CWT)')
  plt.show()

  #*******
  #*******
  #*******
  # PARA WAVELET COMPLEJA
  #En caso de usar wavelet madre **compleja**, abs a todo, menos la ultima fila y columna
  #cwtmatr = np.abs(coef[:-1, :-1])
  # Normaliza los valores de coef entre 0 y 255
  #cwt_normalized = (coef - coef.min()) / (coef.max() - coef.min()) * 255
  # Expande las dimensiones de cwt_normalized para tener la forma (altura, ancho, 3)
  #cwt_rgb = np.stack((cwt_normalized,) * 3, axis=-1)
  # cwt_rgb a tipo entero
  #cwt_rgb = np.uint8(cwt_rgb)
  # Muestra la imagen RGB
  #plt.imshow(cwt_rgb)
  #plt.show()
  #*******
  #*******
  #*******


#####################################################################
############# Aplicacion de ruido y filtro ##########################
#####################################################################

# Directorio donde se almacenaran las senales con ruido
output_folder = "D:/TT/Memoria/waveletycnn/signals_noise"
os.makedirs(output_folder, exist_ok=True)  # Crea la carpeta si no existe

# Iterar sobre el arreglo de diccionario (lugar donde se almacenan las senales de cada individuo)
for item in signals:
    name_file = item['name_file']
    original_signal = np.array(item['signal_vsc'])

    # Se procede a generar 30 senales con ruido y filtro para cada senal original
    for i in range (30):
        # Aplicar ruido y filtro
        filtered_signal = apply_noise_and_filter(original_signal)

        # Crear el nombre del archivo
        file_name = f"{name_file}_ruido{i + 1}.csv"
        file_path = os.path.join(output_folder, file_name)

        # Guardar la senal en el archivo CSV
        df = pd.DataFrame(filtered_signal, columns=["vsc_noise"])
        df.to_csv(file_path, index=False)

print("Señales con ruido generadas y guardadas correctamente.")
