%{
Nombre de funcion: is_power_of_two()
Descripcion: funcion encargada de verificar si la cantidad de muestras de una senal es potencia de 2
Entrada: 
    number [int]: cantidad de muestras de la senal
    signal_name [string]: nombre de la senal que se esta analizando, en este caso la senal de velocidad sanguinea cerebral
Salida: void
%}
function is_power_of_two(number, signal_name)
    % Verificar si el número es potencia de dos
    if mod(log2(number), 1) == 0
        fprintf('El largo de %s es potencia de 2.\n', signal_name);
    else
        fprintf('El largo de %s no es potencia de 2. Por favor, corregir la cantidad de muestras de la señal.\n', signal_name);
    end
end
