function plot_signal_and_scalogram(tms,signal_to_analyze,frq,cfs)
    figure
    subplot(2,1,1)
    plot(tms,signal_to_analyze)
    axis tight
    title("Signal and Scalogram")
    xlabel("Time (s)")
    ylabel("Amplitude")
    subplot(2,1,2)
    surface(tms,frq,abs(cfs))
    axis tight
    shading flat
    xlabel("Time (s)")
    ylabel("Frequency (Hz)")
    set(gca,"yscale","log")


end