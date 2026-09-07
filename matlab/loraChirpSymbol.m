function chirp = loraChirpSymbol(symbolValue, SF, BW, Fs, direction)
% loraChirpSymbol  Generate one LoRa CSS baseband chirp symbol.
%
%   chirp = loraChirpSymbol(symbolValue, SF, BW, Fs, direction)
%
%   symbolValue : integer in [0, 2^SF - 1], the data value encoded by
%                 this symbol's starting frequency (use 0 for a plain
%                 reference up-chirp, e.g. preamble).
%   SF          : spreading factor (7-12)
%   BW          : signal bandwidth in Hz (e.g. 125e3)
%   Fs          : sampling frequency in Hz (Fs >= BW, oversample by
%                 4-8x for smooth chirps, e.g. Fs = 8*BW)
%   direction   : 'up' (data symbols, preamble) or 'down' (sync footer)
%
%   Returns a complex baseband column vector representing one chirp
%   symbol of duration Ts = 2^SF / BW seconds.
%
%   Reference: LoRa CSS instantaneous frequency sweeps linearly across
%   the channel bandwidth once per symbol, starting at a frequency set
%   by the symbol value and wrapping around at +BW/2 (this is the
%   "sawtooth" wrap that distinguishes LoRa chirps from a plain LFM
%   chirp). This matches widely-used open reverse-engineered LoRa PHY
%   models (e.g. gr-lora / "Reversing LoRa" and academic MATLAB
%   reimplementations), since MATLAB has no built-in LoRa PHY block.

    if nargin < 5
        direction = 'up';
    end

    N  = 2^SF;                 % chips per symbol
    Ts = N / BW;                % symbol duration (s)
    numSamples = round(Ts * Fs);
    t = (0:numSamples-1).' / Fs;

    k  = BW / Ts;               % chirp rate (Hz/s), BW swept over Ts
    f0 = -BW/2 + (symbolValue / N) * BW;   % start frequency for this symbol

    if strcmpi(direction, 'down')
        k = -k;                 % down-chirp: frequency decreases
        f0 = BW/2 - (symbolValue / N) * BW;
    end

    % Instantaneous frequency with sawtooth wrap into [-BW/2, BW/2)
    instFreq = mod(f0 + k .* t + BW/2, BW) - BW/2;

    % Integrate frequency to get phase (cumulative trapezoidal), then
    % form the complex baseband chirp.
    phase = 2*pi*cumtrapz(t, instFreq);
    chirp = exp(1j * phase);
end
