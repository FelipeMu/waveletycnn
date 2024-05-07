
%Descripción: Funcion encargada de encontrar el error minimo entre todos
%             los valores que puede tomar VoicesPerOctave [1,48]
function min_error_amor_bump(signal, w)

    error_min=100;
    voices_index = 0;
    for i=1:48
        fb = cwtfilterbank(SignalLength=length(signal),Boundary="periodic", Wavelet=w,SamplingFrequency=5,VoicesPerOctave=i);
        psif = freqz(fb,FrequencyRange="twosided",IncludeLowpass=true);
        [coefs,~,~,scalcfs] = wt(fb,signal);
        xrecAN = icwt(coefs,[],ScalingCoefficients=scalcfs,...
            AnalysisFilterBank=psif);
        xrecAN = xrecAN(:);
        errorAN = get_nmse(signal, xrecAN);
        if errorAN < error_min
            error_min = errorAN;
            voices_index = i;
         
        end
    end     
  
    disp("error mínimo:");
    disp(error_min);
    disp("VoicesPerOctave:");
    disp(voices_index);
end