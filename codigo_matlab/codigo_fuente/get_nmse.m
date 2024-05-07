%{
Nombre de funcion: get_nmse()
Descripcion: funcion encargada de calcular el error NMSE entre la senal original
              y la reconstruida entregado por la Wavelet inversa.
Entrada:
   original_signal [array]: senal PAM original
   predict_signal [array]: senal PAM reconstruida
Salida:
   nmse [float]: error NMSE entre la senal original y la reconstruida
%}
function nmse = get_nmse(original_signal, predict_signal)
    % Calcular el error cuadratico medio normalizado entre la señal original y la reconstruida
    squared_error = (original_signal - predict_signal).^ 2;

    % Calcular la media del error cuadratico
    mean_squared_error = mean(squared_error);

    % Calcular la varianza de la senal original
    variance = var(original_signal);

    % Calcular NMSE (Normalized Mean Squared Error)
    nmse = mean_squared_error / variance;
    
end

