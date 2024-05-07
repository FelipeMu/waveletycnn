function min_error_morse(signal)

    error_min=100;
    voices_index = 0;
    time_index = 0;
    for j=3:120
        for i=1:48
            fb = cwtfilterbank(SignalLength=length(signal),Boundary="periodic", Wavelet="morse",SamplingFrequency=5,VoicesPerOctave=i,TimeBandwidth=j);
            psif = freqz(fb,FrequencyRange="twosided",IncludeLowpass=true);
            [coefs,~,~,scalcfs] = wt(fb,signal);
            xrecAN = icwt(coefs,[],ScalingCoefficients=scalcfs,...
                AnalysisFilterBank=psif);
            xrecAN = xrecAN(:);
            errorAN = get_nmse(signal, xrecAN);
            if errorAN < error_min
                error_min = errorAN;
                voices_index = i;
                time_index = j;
            end
        end     
    end
    disp("error mínimo:");
    disp(error_min);
    disp("VoicesPerOctave:");
    disp(voices_index);
    disp("TimeBandwidth:");
    disp(time_index);

end