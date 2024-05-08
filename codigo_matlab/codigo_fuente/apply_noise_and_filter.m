function apply_noise_and_filter(struct_signals, sampling_freq, len_signals_noises)
    % Carpeta donde se guardaran los nuevos archivos CSV con ruido
    output_folder = 'D:/TT/Memoria/waveletycnn/codigo_matlab/codigo_fuente/signals_noises';
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    
    % Parametros de ruido gaussiano
    cv_inf = 0.05;
    cv_sup = 0.01;
    % Parametros del filtro Butterworth
    order = 8;  % Octavo orden
    nyquist = sampling_freq / 2;  % Frecuencia de Nyquist
    cutoff_freq = 0.25 * nyquist;  % Frecuencia de corte real en Hz
    [b, a] = butter(order, cutoff_freq / nyquist, 'low');  % Coeficientes del filtro
    
    % Recorrer cada archivo CSV en la estructura struct_signals
    num_signals = numel(struct_signals);
    for idx = 1:num_signals
        original_signal = struct_signals(idx).signal_vsc;  % Senal original
    
        % Crear 30 senales con ruido hemodinamico y aplicar el filtro
        for i = 1:len_signals_noises
            % Generar coeficiente de variacion aleatorio entre 5% y 10%
            cv = (cv_inf + (cv_sup - cv_inf) * rand());  % Coeficiente de variacion en porcentaje
            
            % Calcular desviacion estandar del ruido basado en la media de la senal original
            noise_std = mean(original_signal) * cv;
            
            % Generar ruido blanco gaussiano (GWN)
            noise = noise_std * randn(size(original_signal));
            
            % Agregar el ruido a la senal original
            noisy_signal = original_signal + noise;
            
            % Aplicar el filtro Butterworth pasabajos
            filtered_signal = filtfilt(b, a, noisy_signal);
            
            % Guardar la nueva senal en un archivo CSV
            new_file_name = sprintf('%s_ruidoVSC%d.csv', struct_signals(idx).name_file, i);
            output_path = fullfile(output_folder, new_file_name);
            
            % Guardar la senal en el archivo CSV
            writematrix(filtered_signal, output_path);
        end
    end
end