function symbolEst = loraDemodulateSymbol(rxChirp, SF, BW, Fs)
% loraDemodulateSymbol  Recover the integer symbol value from one
% received LoRa chirp using the standard dechirp + FFT method.
%
%   symbolEst = loraDemodulateSymbol(rxChirp, SF, BW, Fs)
%
%   Multiplying a received up-chirp by a reference down-chirp
%   ("dechirping") collapses it to a single tone whose frequency bin
%   (found via FFT) equals the encoded symbol value. This is the
%   standard non-coherent LoRa demodulation technique and is robust to
%   modest frequency/timing offsets, which is why LoRa tolerates
%   cheap, drifting oscillators in the field.

    N = 2^SF;
    refDownChirp = loraChirpSymbol(0, SF, BW, Fs, 'down');

    % Ensure lengths match (guard against off-by-one rounding)
    L = min(numel(rxChirp), numel(refDownChirp));
    dechirped = rxChirp(1:L) .* refDownChirp(1:L);

    % FFT resolved to N bins (one bin per possible symbol value)
    spectrum = abs(fft(dechirped, N));
    [~, peakBin] = max(spectrum);
    symbolEst = peakBin - 1;   % zero-indexed symbol value
end
