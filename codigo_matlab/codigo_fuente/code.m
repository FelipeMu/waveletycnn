%######################################################
%############## PRE-PROCESAMIENTO #####################
%######################################################


% Periodo de muestreo
ts = 0.2; % segundos

% Frecuencia de muestreo
fs = 1.0 / ts; % Hz

% Mostrar el periodo y frecuencia de muestreo
fprintf('Periodo de muestreo de señales: %.2f [seg]\n', ts);
fprintf('Frecuencia de muestreo de señales: %.2f [Hz]\n', fs);

% Arreglos para almacenar datos de señales
pam = []; % PAM: Presión Arterial Media
vsc = []; % VSC: Velocidad Sanguínea Cerebral

% Directorio que contiene archivos CSV
folder_csv = 'D:/TT/Memoria/waveletycnn/signals';

% Listar archivos en el directorio
files_csv = dir(fullfile(folder_csv, '*.csv'));

% Extraer nombres de los archivos
file_names = {files_csv.name};

% Mostrar los nombres de los archivos encontrados
fprintf('Archivos encontrados:\n');
disp(file_names);

%###########################################################################################################
%###########################################################################################################
%###########################################################################################################


% Crear lista de estructuras. cada estructura le corresponde a un individuo
% Directorio donde están los archivos CSV
carpeta_csv = 'D:/TT/Memoria/waveletycnn/codigo_matlab/codigo_fuente/signals';

% Obtener lista de archivos CSV en la carpeta
files_csv = dir(fullfile(carpeta_csv, '*.csv'));


% Estructura de wavelet: AMOR
structure_amor = struct('name_wavelet', 'amor', 'error', 0.0, 'complex_coeffs_amor', [], 'matrix_real', [], 'matrix_imag', [], 'scals_coeffs_amor', [], 'psif_amor', [], 'signal_vsc_rec', []);

% Estructura de wavelet: MORSE
structure_morse = struct('name_wavelet', 'morse', 'error', 0.0, 'signal_vsc_rec', []);

% Estructura de wavelet: BUMP
structure_bump = struct('name_wavelet', 'bump', 'error', 0.0, 'signal_vsc_rec', []);


% Inicializar el arreglo de estructuras con los campos necesarios
num_files = numel(files_csv); % Se almacena la cantidad de archivos csv leidos
signals(num_files) = struct('name_file', '', 'signal_pam', [], 'signal_vsc', [], 'struct_amor', structure_amor, 'struct_morse', structure_morse, 'struct_bump', structure_bump);
% Definir la estructura con cada uno de sus atributos, tomando en cuenta la
% cantidad de archivos encontrados (senales de invididuos) en la carpeta.
for j = 1:num_files
    signals(j) = struct('name_file', '', 'signal_pam', [], 'signal_vsc', [], 'struct_amor', structure_amor, 'struct_morse', structure_morse, 'struct_bump', structure_bump);
end
% Procesar cada archivo CSV
for idx = 1:num_files
    archivo_csv = files_csv(idx).name; % Nombre del archivo CSV
    ruta_archivo = fullfile(carpeta_csv, archivo_csv); % Ruta completa del archivo CSV
    
    % Leer el contenido del archivo CSV
    data = readmatrix(ruta_archivo); % Lee datos del archivo CSV
    
    % Separa las señales PAM y VSC, eliminando la primera fila (suponiendo encabezados)
    pam = data(2:end, 1); % Presión Arterial Media
    vsc = data(2:end, 2); % Velocidad Sanguínea Cerebral
    
    % Convertir a doble precisión y recortar a 1024 puntos
    pam = double(pam(1:min(end, 1024))); % Asegurar que no exceda el tamaño
    vsc = double(vsc(1:min(end, 1024))); 

    % Asignar a la estructura
    signals(idx).name_file = archivo_csv; % Guardar el nombre del archivo
    signals(idx).signal_pam = pam; % Guardar la señal PAM
    signals(idx).signal_vsc = vsc; % Guardar la señal VSC
end

% Mostrar el contenido del arreglo de estructuras
fprintf('\n**** Mostrando el arreglo de estructuras de CSV ****\n');

% Verificar el contenido del arreglo de estructuras
for idx = 1:num_files
    dicc = signals(idx);
    fprintf('Nombre del archivo: %s\n', dicc.name_file);
    fprintf('Señal PAM: %s - N° de instancias: %d\n', mat2str(dicc.signal_pam), numel(dicc.signal_pam));
    fprintf('Señal VSC: %s - N° de instancias: %d\n', mat2str(dicc.signal_vsc), numel(dicc.signal_vsc));
    fprintf('----------------------------------------\n');
end
fprintf('----------------------------------------\n');


for i = 1:num_files
    disp(signals(i)); % Muestra la estructura de cada elemento
end
%###########################################################################################################
%###########################################################################################################
%###########################################################################################################


% Recorrer el arreglo de estructuras 'signals'
for idx = 1:numel(signals)
    dicc = signals(idx); % Acceder a la estructura actual

    % Verificar si las longitudes de las señales PAM y VSC son diferentes
    if numel(dicc.signal_pam) ~= numel(dicc.signal_vsc)
        fprintf('Arreglos PAM y VSC tienen diferente longitud en el archivo %s.\n', dicc.name_file);
    end

    % Mostrar el nombre del archivo y el número de muestras
    fprintf('%s || Número de muestras: %d\n', dicc.name_file, numel(dicc.signal_pam));

    % Verificar si la longitud de la señal PAM es potencia de dos
    is_power_of_two(numel(dicc.signal_pam), 'señal PAM');

    % Verificar si la longitud de la señal VSC es potencia de dos
    is_power_of_two(numel(dicc.signal_vsc), 'señal VSC');

    fprintf('\n'); % Espacio entre salidas para claridad
end

%###########################################################################################################
%###########################################################################################################
%###########################################################################################################


% Aplicar CWT y espectros para cada senal (por ahora a las senales de VSC)
for i = 1:numel(signals)
    s = signals(i);
    disp("Analizando archivo:");
    disp(s.name_file);
    signal_to_analyze = s.signal_vsc; % Senal para analizar
    % WAVELET MADRE CONTINUA A UTILIZAR: Analytic Morlet (Gabor) Wavelet
    fb_amor = cwtfilterbank(SignalLength=length(signal_to_analyze),Boundary="periodic", Wavelet="amor",SamplingFrequency=5,VoicesPerOctave=10); % se obtiene una estructura (banco de filtros)
    psif_amor = freqz(fb_amor,FrequencyRange="twosided",IncludeLowpass=true); % psif_amor: Como cada filtro responde a diferentes frecuencias. ayuda a comprender como se distribuyen las frecuencias a lo largo de mi señal
    signals(i).struct_amor.psif_amor = psif_amor; % Se guarda las respuestas de las frecuencias en el atributo de la estructura
    [coefs_amor,freqs_amor,~,scalcfs_amor] = wt(fb_amor,signal_to_analyze); % se aplica la transformada continua a la senal
    signals(i).struct_amor.complex_coeffs_amor = coefs_amor; % Se guarda la matriz de coeficientes en su respectivo atributo de la estructura
    signals(i).struct_amor.matrix_real = real(coefs_amor); % Se guarda la parte real de la matrix compleja de coeficientes
    signals(i).struct_amor.matrix_imag = imag(coefs_amor); % Se guarda la parte imaginaria de la matrix compleja de coeficientes
    signals(i).struct_amor.scals_coeffs_amor = scalcfs_amor; % Se guarda el vector de escalas en su respectivo atributo de la estructura
    xrecAN_amor = icwt(coefs_amor,[],ScalingCoefficients=scalcfs_amor,AnalysisFilterBank=psif_amor); % se realiza la transformada inversa continua de la senal
    xrecAN_amor = xrecAN_amor(:); % la reconstruccion de la senal se pasa a formato vector columna
    signals(i).struct_amor.signal_vsc_rec = xrecAN_amor;  % Se guarda signal_rec en su respectiva estructura
    errorAN_amor = get_nmse(signal_to_analyze, signals(i).struct_amor.signal_vsc_rec); % se calcula el nmse
    signals(i).struct_amor.error = errorAN_amor; % se almacena el respectivo nmse en la estructura de la senal analizada

    %###########################################################################################
    % Crear vector que representa los tiempos en los que se toma una muestra
    tms = (0:numel(signal_to_analyze)-1)/fs;
    % Llamada a funcion para mostrar grafica de la senal y su respectivo
    % escalograma
    plot_signal_and_scalogram(tms, signal_to_analyze, freqs_amor, coefs_amor, 'amor',s.name_file)
 end 

%###########################################################################################################
%###########################################################################################################
%###########################################################################################################


% Ejemplo de cómo crear una figura con múltiples secciones para comparar señales y mostrar errores

% Crear una nueva figura
figure;
% Crear subplots para la señal original y las reconstrucciones con amor, morse, bump
% y mostrar el error NMSE en el costado
for i = 1:num_files
    s = signals(i); % Obtenemos la estructura correspondiente a la señal actual

    % Definir el índice base para los subplots (para separar las señales)
    base_idx = (i - 1) * 2;

    % Crear subplots para comparar las señales originales con las reconstruidas por amor
    subplot(num_files, 3, base_idx + 1);
    hold on;
    plot(s.signal_vsc, 'b'); % Señal original
    plot(s.struct_amor.signal_vsc_rec, 'r--'); % Señal reconstruida por amor
    title(sprintf('Señal VSC vs Amor (NMSE: %.2e) [%s]', s.struct_amor.error, s.name_file));
    xlabel('Tiempo');
    ylabel('Amplitud');
    legend('Original', 'Reconstruida (amor)');
    hold off;
end

%###########################################################################################################
%###########################################################################################################
%###########################################################################################################

% BUSQUEDA EL MINIMO NMSE PARA CADA WAVELET, TENIENDO EN CUENTA LOS
% HIPERPARAMETROS QUE UTILIZAN:
% Para encontrar el minimo error, se procede a utilizar las siguientes
% funciones. En ellas se realizan todas las combinaciones posibles de los
% hiperparametros TimeBandwidth y VoicesPerOctave para encontrar el error
% minimo de cada wavelet madre
min_error_amor_bump(signals(1).signal_vsc, "bump"); % para wavelet AMOR y BUMP
min_error_morse(signals(1).signal_vsc); % para wavelet MORSE





%###########################################################################################################
%####################### Aplicacion de ruido Gaussiano y filtro Butterworth ################################
%###########################################################################################################

% Se procede aplicar ruido gaussiano con un coeficiente de variacion entre
% [5%, 10%], y posteriormente un filtro Butterworth de octavo orden, con
% frecuencia de corte de 0.25 Hz.
apply_noise_and_filter(signals, fs, 30);







%###########################################################################################################
%############################ Red profunda [U-Net] Train and Validation ####################################
%###########################################################################################################

