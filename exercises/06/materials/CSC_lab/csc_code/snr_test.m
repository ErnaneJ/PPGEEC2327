function snr=snr_test(signal,noise)

% to calculate the SNR of the signal and noise
% by 10*log10(||siginal||_2/||noise||_2)

snr=10*log10(sum(signal(:).^2)/sum(noise(:).^2));
end