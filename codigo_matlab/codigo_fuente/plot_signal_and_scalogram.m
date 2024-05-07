function plot_signal_and_scalogram(tms,signal_to_analyze,frq,cfs, name_wavelet, file)
    figure
    subplot(2,1,1)
    plot(tms,signal_to_analyze)
    axis tight
    title(sprintf("Archivo: %s - Señal y Escalograma - %s", file, name_wavelet)); 
    xlabel("Tiempo (s)")
    ylabel("Amplitud")
    subplot(2,1,2)
    surface(tms,frq,abs(cfs))
    axis tight
    shading flat
    xlabel("Tiempo (s)")
    ylabel("Frecuencia (Hz)")
    set(gca,"yscale","log")
end