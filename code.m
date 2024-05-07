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
    disp("Estamos analizando el archivo:");
    disp(s.name_file);
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
           
                [coefs_amor, freqs_amor] = cwt(signal_to_analyze, 'amor', fs); % Aplicar CWT

                % Observar el rango de periodos
                min_freq_amor = min(freqs_amor); % Periodo minimo
                max_freq_amor = max(freqs_amor); % Periodo maximo
                
                % Definir el rango de periodos para la reconstruccion con icwt
                freqsrange_amor = [min_freq_amor max_freq_amor]; % Utilizar todo el rango    
                %reconstructed_signal_amor = icwt(coefs_amor, 'amor', freqs_amor, freqsrange_amor,'ScalingCoefficients',); % Aplicar icwt
                reconstructed_signal_amor = icwt(coefs_amor, 'amor', freqs_amor, freqsrange_amor, SignalMean=mean(signal_to_analyze)); % Aplicar icwt
                reconstructed_signal_amor = reconstructed_signal_amor(:); % transformar a vector columna
                signals(i).struct_amor.signal_vsc_rec = reconstructed_signal_amor;  % Se guarda signal_rec en su respectiva estructura
                %###########################################################################################
                % Se procede a calcular el NMSE:
                nmse_amor = get_nmse(signals(i).signal_vsc, signals(i).struct_amor.signal_vsc_rec);
                % Se asigna el NMSE a la respectiva estructura de la
                % wavelet madre:
                signals(i).struct_amor.error = nmse_amor;
                %###########################################################################################
                % Crear vector que representa los tiempos en los que se toma una muestra
                tms = (0:numel(signal_to_analyze)-1)/fs;
                % Llamada a funcion para mostrar grafica de la senal y su respectivo
                % escalograma
                plot_signal_and_scalogram(tms, signal_to_analyze, freqs_amor, coefs_amor, 'amor',s.name_file)

            
                            
                % Aplicar la Transformada de Fourier
                n = length(signal_to_analyze); % Número total de muestras
                y = fft(signal_to_analyze); % Transformada de Fourier
                
                % Calcular el vector de frecuencias
                f = (0:n-1) * (fs / n); % Frecuencias correspondientes a cada componente del FFT
                
                % Calcular la magnitud del FFT (solo la mitad debido a la simetría)
                magnitude = abs(y) / n; % Normalizar la magnitud
                magnitude = magnitude(1:floor(n/2)); % Tomar la mitad debido a la simetría
                f = f(1:floor(n/2)); % Frecuencias correspondientes a la magnitud
                
                % Visualizar el espectro de frecuencias
                figure;
                plot(f, magnitude); % Graficar magnitud vs frecuencia
                xlabel('Frecuencia (Hz)');
                ylabel('Magnitud');
                title('Transformada de Fourier de la señal');


            case 'morse'
            
                %***[coefs_morse, freqs_morse] = cwt(signal_to_analyze, 'morse', fs); % Aplicar CWT
                [coefs_bump, freqs_morse] = cwt(signal_to_analyze, 'morse', fs, VoicesPerOctave=48);
                % Observar el rango de periodos
                min_freq_morse = min(freqs_morse); % Periodo minimo
                max_freq_morse = max(freqs_morse); % Periodo maximo
                
                % Definir el rango de periodos para la reconstruccion con icwt
                freqsrange_morse = [min_freq_morse max_freq_morse]; % Utilizar todo el rango 

                [minfreq,maxfreq] = cwtfreqbounds(length(signal_to_analyze),fs);


                %***reconstructed_signal_morse = icwt(coefs_morse, 'morse',SignalMean=mean(signal_to_analyze)); % Aplicar icwt
                reconstructed_signal_morse = icwt(coefs_bump, 'morse', SignalMean=mean(signal_to_analyze), VoicesPerOctave=48); % Aplicar icwt
                reconstructed_signal_morse = reconstructed_signal_morse(:); % transformar a vector columna
                signals(i).struct_morse.signal_vsc_rec = reconstructed_signal_morse; % Se guarda signal_rec en su respectiva estructura
                %###########################################################################################
                % Se procede a calcular el NMSE:
                nmse_morse = get_nmse(signals(i).signal_vsc, reconstructed_signal_morse);
                % Se asigna el NMSE a la respectiva estructura de la
                % wavelet madre:
                signals(i).struct_morse.error = nmse_morse;
                %###########################################################################################
                % Crear vector que representa los tiempos en los que se toma una muestra
                tms = (0:numel(signal_to_analyze)-1)/fs;
                % Llamada a funcion para mostrar grafica de la senal y su respectivo
                % escalograma
                plot_signal_and_scalogram(tms, signal_to_analyze, freqs_morse, coefs_bump, 'morse', s.name_file)
            case 'bump'
            
                [coefs_bump, freqs_bump] = cwt(signal_to_analyze, 'bump',fs); % Aplicar CWT
                reconstructed_signal_bump = icwt(coefs_bump, 'bump',SignalMean=mean(signal_to_analyze)); % Aplicar icwt
                reconstructed_signal_bump = reconstructed_signal_bump(:); % tramsformar a vector columna
                signals(i).struct_bump.signal_vsc_rec = reconstructed_signal_bump;  % Se guarda signal_rec en su respectiva estructura
                %###########################################################################################
                % Se procede a calcular el NMSE:
                nmse_bump = get_nmse(signals(i).signal_vsc, reconstructed_signal_bump);
                % Se asigna el NMSE a la respectiva estructura de la
                % wavelet madre:
                signals(i).struct_bump.error = nmse_bump;
                %###########################################################################################
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



%valor = 0.12345678;
%notacion_cientifica = sprintf('%.2e', valor); % '%.2e' muestra el número con dos decimales en notación científica
%disp(notacion_cientifica); % Muestra el valor en notación científica


% Ejemplo de cómo crear una figura con múltiples secciones para comparar señales y mostrar errores

% Crear una nueva figura
figure;

% Definir el número de filas y columnas para los subplots
num_signals = 2; % En este caso, estamos trabajando con signal(1) y signal(2)

% Crear subplots para la señal original y las reconstrucciones con amor, morse, bump
% y mostrar el error NMSE en el costado
for i = 1:num_signals
    s = signals(i); % Obtenemos la estructura correspondiente a la señal actual

    % Definir el índice base para los subplots (para separar las señales)
    base_idx = (i - 1) * 3;

    % Crear subplots para comparar las señales originales con las reconstruidas por amor
    subplot(num_signals, 3, base_idx + 1);
    hold on;
    plot(s.signal_vsc, 'b'); % Señal original
    plot(s.struct_amor.signal_vsc_rec, 'r--'); % Señal reconstruida por amor
    title(sprintf('Señal VSC vs Amor (NMSE: %.2e) [%s]', s.struct_amor.error, s.name_file));
    xlabel('Tiempo');
    ylabel('Amplitud');
    legend('Original', 'Reconstruida (amor)');
    hold off;

    % Crear subplots para comparar las señales originales con las reconstruidas por morse
    subplot(num_signals, 3, base_idx + 2);
    hold on;
    plot(s.signal_vsc, 'b'); % Señal original
    plot(s.struct_morse.signal_vsc_rec, 'r--'); % Señal reconstruida por morse
    title(sprintf('Señal VSC vs Morse (NMSE: %.2e) [%s]', s.struct_morse.error, s.name_file));
    xlabel('Tiempo');
    ylabel('Amplitud');
    legend('Original', 'Reconstruida (morse)');
    hold off;

    % Crear subplots para comparar las señales originales con las reconstruidas por bump
    subplot(num_signals, 3, base_idx + 3);
    hold on;
    plot(s.signal_vsc, 'b'); % Señal original
    plot(s.struct_bump.signal_vsc_rec, 'r--'); % Señal reconstruida por bump
    title(sprintf('Señal VSC vs Bump (NMSE: %.2e) [%s]', s.struct_bump.error, s.name_file));
    xlabel('Tiempo');
    ylabel('Amplitud');
    legend('Original', 'Reconstruida (bump)');
    hold off;
end





% UTILIZANDO cwtmag2sig()
%{
[CFS,~,~,~,scalcfs] = cwt(signals(1).signal_vsc,ExtendSignal=false);
xrec = cwtmag2sig(abs(CFS),...
 Display=true,ScalingCoefficients=scalcfs);
error = get_nmse(signals(1).signal_vsc, xrec);
fs = 5; % Frecuencia de muestreo de 5 Hz
n = 1024; % Número total de muestras
ts = 0:1/fs:(n-1)/fs; % Vector de tiempo, desde 0 hasta la duración total
plot(ts,signals(1).signal_vsc,ts,xrec,"--")
xlabel("Time (s)")
ylabel("Amplitude")
legend("Original","Reconstructed")
%}

%##############
% WAVELET AMOR
%##############
% Estructura que almacena caracteristicas de la wavelet a aplicar:
fb_amor = cwtfilterbank(SignalLength=length(signals(2).signal_vsc),Boundary="periodic", Wavelet="amor",SamplingFrequency=5,VoicesPerOctave=16);
psif_amor = freqz(fb_amor,FrequencyRange="twosided",IncludeLowpass=true);
[coefs_amor,~,~,scalcfs_amor] = wt(fb_amor,signals(2).signal_vsc);
xrecAN_amor = icwt(coefs_amor,[],ScalingCoefficients=scalcfs_amor,...
    AnalysisFilterBank=psif_amor);
xrecAN_amor = xrecAN_amor(:);
errorAN_amor = get_nmse(signals(2).signal_vsc, xrecAN_amor);

fs = 5; % Frecuencia de muestreo de 5 Hz
n = 1024; % Número total de muestras
ts = 0:1/fs:(n-1)/fs; % Vector de tiempo, desde 0 hasta la duración total
plot(ts,signals(2).signal_vsc,ts,xrecAN_amor,"--")
xlabel("Time (s)")
ylabel("Amplitude")
legend("Original","Reconstructed")

%##############
% WAVELET MORSE
%##############
% Estructura que almacena caracteristicas de la wavelet a aplicar:
fb_morse = cwtfilterbank(SignalLength=length(signals(2).signal_vsc),Boundary="periodic",SamplingFrequency=5,TimeBandwidth=60,VoicesPerOctave=7);
psif_morse = freqz(fb_morse,FrequencyRange="twosided",IncludeLowpass=true);
[coefs_morse,~,~,scalcfs_morse] = wt(fb_morse,signals(2).signal_vsc);
xrecAN_morse = icwt(coefs_morse,[],ScalingCoefficients=scalcfs_morse,...
    AnalysisFilterBank=psif_morse);
xrecAN_morse = xrecAN_morse(:);
errorAN_morse = get_nmse(signals(2).signal_vsc, xrecAN_morse);

fs = 5; % Frecuencia de muestreo de 5 Hz
n = 1024; % Número total de muestras
ts = 0:1/fs:(n-1)/fs; % Vector de tiempo, desde 0 hasta la duración total
plot(ts,signals(2).signal_vsc,ts,xrecAN_morse,"--")
xlabel("Time (s)")
ylabel("Amplitude")
legend("Original","Reconstructed")


%##############
% WAVELET BUMP
%##############
% Estructura que almacena caracteristicas de la wavelet a aplicar:
fb_bump = cwtfilterbank(SignalLength=length(signals(2).signal_vsc),Boundary="periodic", Wavelet="bump", SamplingFrequency=5);
psif_bump = freqz(fb_bump,FrequencyRange="twosided",IncludeLowpass=true);
[coefs_bump,~,~,scalcfs_bump] = wt(fb_bump,signals(2).signal_vsc);
xrecAN_bump = icwt(coefs_bump,[],ScalingCoefficients=scalcfs_bump,...
    AnalysisFilterBank=psif_bump);
xrecAN_bump = xrecAN_bump(:);
errorAN_bump = get_nmse(signals(2).signal_vsc, xrecAN_bump);

fs = 5; % Frecuencia de muestreo de 5 Hz
n = 1024; % Número total de muestras
ts = 0:1/fs:(n-1)/fs; % Vector de tiempo, desde 0 hasta la duración total
plot(ts,signals(2).signal_vsc,ts,xrecAN_bump,"--")
xlabel("Time (s)")
ylabel("Amplitude")
legend("Original","Reconstructed")



% Para encontrar el minimo error, se procede a utilizar las siguientes
% funciones. En ellas se realizan todas las combinaciones posibles de los
% hiperparametros TimeBandwidth y VoicesPerOctave para encontrar el error
% minimo de cada wavelet madre
min_error_amor_bump(signals(1).signal_vsc, "bump");
min_error_morse();