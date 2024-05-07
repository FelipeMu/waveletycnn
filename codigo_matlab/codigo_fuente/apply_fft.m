function apply_fft(signal, ts)
    % Paso 1: Aplicar la Transformada de Fourier
    transformada = fft(signal);
    
    % Paso 2: Calcular las frecuencias correspondientes
    frecuencias = fftfreq(length(signal), ts);
    
    % Determinar la frecuencia máxima y mínima
    freq_max = max(frecuencias);
    freq_min = 2.50; % Valor predeterminado
    
    abs_freq = abs(frecuencias);
    for f = abs_freq
        if f <= freq_min && f ~= 0
            freq_min = f;
        end
    end
    
    % Imprimir las frecuencias mínimas y máximas
    fprintf('La frecuencia mínima es: %.2f Hz\n', freq_min);
    fprintf('La frecuencia máxima es: %.2f Hz\n', freq_max);
    
    % Paso 3: Graficar amplitud vs. frecuencia
    figure;
    plot(frecuencias, abs(transformada));
    xlabel('Frecuencia (Hz)');
    ylabel('Amplitud');
    title('Transformada de Fourier');
    grid on;
    
    % Mostrar la frecuencia máxima en el gráfico
    text(0.95, 0.95, sprintf('freq_max: %.2f Hz', freq_max), ...
        'Units', 'normalized', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
    
    % Mostrar el gráfico
    show;
end