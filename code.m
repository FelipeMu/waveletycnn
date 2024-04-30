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

%############################################################
%############################################################
%############################################################

% Crear lista de estructuras. cada estructura le corresponde a un individuo
% Directorio donde están los archivos CSV
carpeta_csv = 'D:/TT/Memoria/waveletycnn/signals';

% Obtener lista de archivos CSV en la carpeta
files_csv = dir(fullfile(carpeta_csv, '*.csv'));


% Estructura de wavelet: AMOR
structure_amor = struct('name_wavelet', 'amor', 'error', 0.0, 'signal_vsc_rec', []);

% Estructura de wavelet: MORSE
structure_morse = struct('name_wavelet', 'morse', 'error', 0.0, 'signal_vsc_rec', []);

% Estructura de wavelet: BUMP
structure_bump = struct('name_wavelet', 'bump', 'error', 0.0, 'signal_vsc_rec', []);



% Inicializar el arreglo de estructuras con los campos necesarios
num_files = numel(files_csv);
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

%############################################################
%############################################################
%############################################################

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

%##########################################################################
%##########################################################################
%##########################################################################

% Se define arreglo que almacena las wavelet continuas disponibles
wavelets_cwt = ["amor", "morse", "bump"];

% Aplicar CWT y espectros para cada senal (por ahora a las senales de VSC)
for i = 1:numel(signals)
    s = signals(i);
    for w = 1:numel(wavelets_cwt)
        % Paso 1: Elegir la wavelet
        wname = wavelets_cwt(w); % Wavelet 
        signal_to_analyze = s.signal_vsc; % Señal para analizar
        

        
        % Paso 2: Preparar las entradas para FFT y CWT
        %n = length(s.signal_vsc); % Cantidad de muestras
        %t = linspace(0, n * ts, n); % Vector de tiempo
        % Paso 3: Analizar FFT
        %apply_fft(signal_to_analyze, ts); % Llamar a la función FFT
        %scales = 1:512; % Escalas para la CWT. La funcion cwt calcula
        % la escala min y max de la forma mas apropiada
        
        
        % Guardar la reconstruccion en su respectiva struct dependiendo de
        % la wavelet madre que se esta ejecutando.
        switch wname
            case 'amor'
           
                [coefs_amor, freqs_amor] = cwt(signal_to_analyze, 'amor'); % Aplicar CWT
                reconstructed_signal_amor = icwt(coefs_amor, 'amor',SignalMean=mean(signal_to_analyze));
                %s.struct_amor.signal_vsc_rec = reconstructed_signal_amor;
                signals(i).struct_amor.signal_vsc_rec = reconstructed_signal_amor;

                % Crear vector que representa los tiempos en los que se toma una muestra
                tms = (0:numel(signal_to_analyze)-1)/fs;
                % Llamada a funcion para mostrar grafica de la senal y su respectivo
                % escalograma
                plot_signal_and_scalogram(tms, signal_to_analyze, freqs_amor, coefs_amor, 'amor',s.name_file)
            case 'morse'
            
               [coefs_morse, freqs_morse] = cwt(signal_to_analyze, 'morse'); % Aplicar CWT
               reconstructed_signal_morse = icwt(coefs_morse, 'morse',SignalMean=mean(signal_to_analyze));
               %s.struct_morse.signal_vsc_rec = reconstructed_signal_morse;
               signals(i).struct_morse.signal_vsc_rec = reconstructed_signal_morse;

               % Crear vector que representa los tiempos en los que se toma una muestra
                tms = (0:numel(signal_to_analyze)-1)/fs;
                % Llamada a funcion para mostrar grafica de la senal y su respectivo
                % escalograma
                plot_signal_and_scalogram(tms, signal_to_analyze, freqs_morse, coefs_morse, 'morse', s.name_file)
            case 'bump'
            
                [coefs_bump, freqs_bump] = cwt(signal_to_analyze, 'bump'); % Aplicar CWT
                reconstructed_signal_bump = icwt(coefs_bump, 'bump',SignalMean=mean(signal_to_analyze));
                %s.struct_bump.signal_vsc_rec = reconstructed_signal_bump;
                signals(i).struct_bump.signal_vsc_rec = reconstructed_signal_bump;

                % Crear vector que representa los tiempos en los que se toma una muestra
                tms = (0:numel(signal_to_analyze)-1)/fs;
                % Llamada a funcion para mostrar grafica de la senal y su respectivo
                % escalograma
                plot_signal_and_scalogram(tms, signal_to_analyze, freqs_bump, coefs_bump, 'bump', s.name_file)
            otherwise
                fprintf('No se ha encontrado conincidencia con dicha wavelet')
        end
    end
 end 

%#############################################################################
%#############################################################################
%#############################################################################


%[cfs1, frq1] = cwt(signals(1).signal_vsc, 'bump'); % Aplicar CWT
%reconstructed_signal = icwt(cfs1, 'bump',SignalMean=mean(signals(1).signal_vsc));
%Fs = 5.0;
%dt = 1/Fs;
%T = 0:dt:numel(signals(1).signal_vsc)*dt-dt;
%plot(T,signals(1).signal_vsc)
%xlabel("Seconds")
%ylabel("Amplitude")
%hold on
%plot(T,reconstructed_signal,"r")
%hold off
%axis tight
%legend("Original","Reconstruction")