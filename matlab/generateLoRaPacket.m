function [txSignal, symbolStream] = generateLoRaPacket(payloadBytes, SF, BW, Fs)
% generateLoRaPacket  Assemble a simplified LoRa CSS packet waveform.
%
%   [txSignal, symbolStream] = generateLoRaPacket(payloadBytes, SF, BW, Fs)
%
%   payloadBytes : uint8 row vector, the message to send (e.g. a
%                  disaster alert / beacon payload)
%   SF, BW, Fs   : spreading factor, bandwidth (Hz), sample rate (Hz)
%
%   Returns the complex baseband tx waveform and the sequence of
%   integer symbol values that were encoded (useful later for
%   computing symbol error rate after demodulation).
%
%   NOTE: this is a research/simulation-level PHY model — it captures
%   the CSS modulation, preamble, and sync footer that dominate
%   receiver performance, but omits full LoRaWAN-spec whitening,
%   Hamming FEC, and interleaving. Add those later if you need
%   bit-exact conformance; for link-quality/ML-dataset generation the
%   simplified model is sufficient and much easier to validate.

    N = 2^SF;

    % --- Preamble: repeated reference up-chirps (symbol 0) ---
    numPreamble = 8;
    preambleSymbols = zeros(1, numPreamble);
    preambleChirps = cell(1, numPreamble);
    for i = 1:numPreamble
        preambleChirps{i} = loraChirpSymbol(0, SF, BW, Fs, 'up');
    end

    % --- Sync word: 2 up-chirps carrying a fixed sync value ---
    syncValue = round(0.25 * N);   % arbitrary fixed public sync symbol
    syncChirps = {loraChirpSymbol(syncValue, SF, BW, Fs, 'up'), ...
                  loraChirpSymbol(syncValue, SF, BW, Fs, 'up')};

    % --- Start-of-frame delimiter: 2.25 down-chirps ---
    downChirp = loraChirpSymbol(0, SF, BW, Fs, 'down');
    quarterLen = round(0.25 * numel(downChirp));
    sfd = [downChirp; downChirp; downChirp(1:quarterLen)];

    % --- Payload: map each byte to one symbol (0..255) modulo N ---
    % For SF < 8, N < 256 so we wrap; for SF >= 8 this maps directly.
    payloadSymbols = mod(double(payloadBytes), N);
    payloadChirps = cell(1, numel(payloadSymbols));
    for i = 1:numel(payloadSymbols)
        payloadChirps{i} = loraChirpSymbol(payloadSymbols(i), SF, BW, Fs, 'up');
    end

    % --- Concatenate everything into the tx waveform ---
    txSignal = cat(1, preambleChirps{:}, syncChirps{:}, sfd, payloadChirps{:});
    symbolStream = payloadSymbols;
end
