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
folder_csv = 'D:/TT/Memoria/waveletycnn/codigo_matlab_codigo_fuente/signals';

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
    pam = data(:, 1); % Presion Arterial Media
    vsc = data(:, 2); % Velocidad Sanguinea Cerebral
    
    % Convertir a double y recortar a 1024 puntos
    pam = double(pam(1:min(end, 1024))); % Asegurar que no exceda el tamano
    vsc = double(vsc(1:min(end, 1024))); 

    % Asignar a la estructura
    signals(idx).name_file = archivo_csv; % Guardar el nombre del archivo
    signals(idx).signal_pam = pam; % Guardar la senal PAM
    signals(idx).signal_vsc = vsc; % Guardar la senal VSC
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
    fb_amor = cwtfilterbank(SignalLength=length(signal_to_analyze),Boundary="periodic", Wavelet="amor",SamplingFrequency=5,VoicesPerOctave=5); % se obtiene una estructura (banco de filtros)
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


% Ejemplo de como crear una figura con multiples secciones para comparar senales y mostrar errores

% Crear una nueva figura
figure;
% Crear subplots para la señal original y las reconstrucciones con amor, morse, bump
% y mostrar el error NMSE en el costado
for i = 1:num_files
    s = signals(i); % Obtenemos la estructura correspondiente a la senal actual

    % Definir el índice base para los subplots (para separar las señales)
    base_idx = (i - 1) * 2;

    % Crear subplots para comparar las senales originales con las reconstruidas por amor
    subplot(num_files, 3, base_idx + 1);
    hold on;
    plot(s.signal_vsc, 'b'); % Senal original
    plot(s.struct_amor.signal_vsc_rec, 'r--'); % Senal reconstruida por amor
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
% frecuencia de corte de 0.25 Hz. (estructura signals, frecuencia de muestreo, [# cantidad] de senales con ruido)
apply_noise_and_filter(signals, fs, 30);







%###########################################################################################################
%############################ Preparacion de inputs para la red ############################################
%###########################################################################################################


% Por cada senal pam & vsc se debe aplicar CWT, esto con el fin de
% obtener los inputs para el entrenamiento de la red.

path_1 = 'D:/TT/Memoria/waveletycnn/codigo_matlab/codigo_fuente/signals_noises';
% Obtener los nombres de las carpetas dentro del directorio
folder_structs = dir(path_1);
folder_names = {folder_structs([folder_structs.isdir]).name}; % se obtiene los nombres de las carpetas
% Eliminar los nombres '.' y '..' que representan el directorio actual y el directorio padre
folder_names = setdiff(folder_names, {'.', '..'});

% Elegir al sujeto de prueba
file_person = folder_names(1); % se tiene el nombre de la carpeta: ejemplo G2x001

file_pam = fullfile(path_1, file_person); % Se concatena al path_1 (path general), el nombre de la carpeta del sujeto


% Obtener los nombres de las carpetas dentro del directorio del sujeto que
% se va a estudiar
folder_structs2 = dir(file_pam{1}); % Se obtiene la ruta2 como string
folder_names2 = {folder_structs2([folder_structs2.isdir]).name}; % se obtiene los nombres de las carpetas
% Eliminar los nombres '.' y '..' que representan el directorio actual y el directorio padre
folder_names2 = setdiff(folder_names2, {'.', '..'}); % Se almacenan los nombres de PAMnoises y VSCnoises

% Se obtiene el directorio de la carpeta que almacena las senales PAM con
% ruido:
path_pam_noises = fullfile(file_pam{1}, folder_names2{1}); % directorio de PAMnoises

% Se obtiene el directorio de la carpeta que almacena las senales VSC con
% ruido:
path_vsc_noises = fullfile(file_pam{1}, folder_names2{2}); % directorio de PAMnoises


% Ahora se deben extraer todos los archivos .csv del directorio
% path_pam_noises:

% Obtener lista de archivos CSV en la carpeta de PAMnoises
pam_noises_csv = dir(fullfile(path_pam_noises, '*.csv'));

% Obtener lista de archivos CSV en la carpeta de VSCnoises
vsc_noises_csv = dir(fullfile(path_vsc_noises, '*.csv'));

% Cantidad de csvs encontrados en el directorio path_pam_noises
num_csv = numel(pam_noises_csv); % Se almacena la cantidad de archivos csv leidos



% Crear estructura que guardara cada par de PAM y VSC con ruido
struct_noises(num_csv) = struct('name_signal', '', 'pam_noise', [], 'matrix_complex_pam', [], 'scalscfs_pam_noise', [], 'psif_pam_noise', [],  'vsc_noise', [], 'matrix_complex_vsc', [], 'scalscfs_vsc_noise', [], 'psif_vsc_noise', []);

% Se crean tantas instancias de la estructura como archivos csv encontrados
% en la carpeta
for j = 1:num_csv
    struct_noises(j) = struct('name_signal', '', 'pam_noise', [], 'matrix_complex_pam', [], 'scalscfs_pam_noise', [], 'psif_pam_noise', [],  'vsc_noise', [], 'matrix_complex_vsc', [], 'scalscfs_vsc_noise', [], 'psif_vsc_noise', []);
end


% *********************************************************************
% Bucle para almacenar las senales con ruido en una estructura unica:
% *********************************************************************
for index = 1:num_csv

    file2_csv = pam_noises_csv(index).name; % Nombre del archivo  PAMnoises
    file2_csv_vsc = vsc_noises_csv(index).name; % Nombre del archivo VSCnoises
    
    path_pam_file2 = fullfile(path_pam_noises, file2_csv); % Ruta completa de la carpeta PAMnoises
    path_vsc_file2 = fullfile(path_vsc_noises, file2_csv_vsc); % Ruta completa de la carpeta VSCnoises

    % Leer el contenido del archivo CSV
    data2_pam_noises = readmatrix(path_pam_file2); % Lee datos del archivo CSV
    % Leer el contenido del archivo CSV
    data2_vsc_noises = readmatrix(path_vsc_file2); % Lee datos del archivo CSV
    
    % Separa las señales PAM y VSC, eliminando la primera fila (suponiendo encabezados)
    pam_noise = data2_pam_noises(:, 1); % Presion Arterial Media con ruido
    vsc_noise = data2_vsc_noises(:, 1); % Velocidad Sanguinea Cerebral con ruido

    % Asignar a la estructura
    struct_noises(index).name_signal = ['Ruido', num2str(index)]; % Guardar el nombre del archivo
    struct_noises(index).pam_noise = pam_noise; % Guardar la senal PAM con ruido
    struct_noises(index).vsc_noise = vsc_noise; % Guardar la senal VSC con ruido

    %########################################
    %######## Aplicacion de CWT #############
    %########################################

    % WAVELET MADRE CONTINUA A UTILIZAR: Analytic Morlet (Gabor) Wavelet
    
    % [PAM NOISE - CWT]
    filters_bank_pam_noise = cwtfilterbank(SignalLength=length(struct_noises(index).pam_noise),Boundary="periodic", Wavelet="amor",SamplingFrequency=5,VoicesPerOctave=5); % se obtiene una estructura (banco de filtros)
    psif_pam_noise = freqz(filters_bank_pam_noise,FrequencyRange="twosided",IncludeLowpass=true); % psif_amor: Como cada filtro responde a diferentes frecuencias. ayuda a comprender como se distribuyen las frecuencias a lo largo de mi señal
    [coefs_pam_noise,freqs_pam_noise,~,scalcfs_pam_noise] = wt(filters_bank_pam_noise,struct_noises(index).pam_noise); % se aplica la transformada continua a la senal
    
    % [VSC NOISE - CWT]
    filters_bank_vsc_noise = cwtfilterbank(SignalLength=length(struct_noises(index).vsc_noise),Boundary="periodic", Wavelet="amor",SamplingFrequency=5,VoicesPerOctave=5); % se obtiene una estructura (banco de filtros)
    psif_vsc_noise = freqz(filters_bank_vsc_noise,FrequencyRange="twosided",IncludeLowpass=true); % psif_amor: Como cada filtro responde a diferentes frecuencias. ayuda a comprender como se distribuyen las frecuencias a lo largo de mi señal
    [coefs_vsc_noise,freqs_vsc_noise,~,scalcfs_vsc_noise] = wt(filters_bank_vsc_noise,struct_noises(index).vsc_noise); % se aplica la transformada continua a la senal
    
    
    % Almacenando nueva informacion en la respectiva estructura de senales
    % con ruido:
    
    % Almacenando coeficientes (matriz compleja)
    struct_noises(index).matrix_complex_pam = coefs_pam_noise; % pam
    struct_noises(index).matrix_complex_vsc = coefs_vsc_noise; % vsc

    % Almacenando escalas de coeficientes (vector 1D real en fila, largo 1024)
    struct_noises(index).scalscfs_pam_noise = scalcfs_pam_noise; % pam
    struct_noises(index).scalscfs_vsc_noise = scalcfs_vsc_noise; % vsc

    % Almacenando respuestas de filtros (matriz real 30x1024)
    struct_noises(index).psif_pam_noise = psif_pam_noise; % pam
    struct_noises(index).psif_vsc_noise = psif_vsc_noise; % vsc

end



% Almacenar matrices complejas pam y vsc en carpetas especificas para 
% luego trabajar con la red profunda en python. Para ello se importan 
% las matrices en formato.mat y luego en python se utiliza un script
% para transformar dicho formato a npy.

% Directorios para guardar los archivos .mat
pam_dir = 'D:/TT/Memoria/waveletycnn/codigo_matlab/codigo_fuente/matrices_complejas_pam_mat';
vsc_dir = 'D:/TT/Memoria/waveletycnn/codigo_matlab/codigo_fuente/matrices_complejas_vsc_mat';

% Crear los directorios si no existen
if ~exist(pam_dir, 'dir')
    mkdir(pam_dir);
end
if ~exist(vsc_dir, 'dir')
    mkdir(vsc_dir);
end

% Guardar las matrices complejas en archivos .mat
for i = 1:num_csv
    % Guardar matriz_complex_pam
    matrix_complex_pam = struct_noises(i).matrix_complex_pam;
    save(fullfile(pam_dir, sprintf('matrix_complex_pam_noise_%d.mat', i)), 'matrix_complex_pam');
    
    % Guardar matriz_complex_vsc
    matrix_complex_vsc = struct_noises(i).matrix_complex_vsc;
    save(fullfile(vsc_dir, sprintf('matrix_complex_vsc_noise_%d.mat', i)), 'matrix_complex_vsc');
end
