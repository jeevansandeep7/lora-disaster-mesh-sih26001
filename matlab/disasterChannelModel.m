function rxSignal = disasterChannelModel(txSignal, Fs, params)
% disasterChannelModel  Apply disaster-scenario channel impairments to
% a LoRa baseband waveform.
%
%   rxSignal = disasterChannelModel(txSignal, Fs, params)
%
%   params fields (all optional, sensible defaults applied):
%     .SNR_dB        - target SNR in dB after fading (default 5)
%     .fadingType    - 'none' | 'rician' | 'rayleigh'  (default 'rician')
%     .kFactorRician - Rician K-factor in dB, higher = more LOS,
%                      lower = more debris/NLOS scattering (default 3)
%     .maxDopplerHz  - Doppler spread from node mobility (default 5)
%     .freqOffsetHz  - carrier frequency offset from cheap oscillators
%                      / battery-drift (default 0)
%     .battteryScale - amplitude scale in (0,1], models a node
%                      transmitting on a depleted battery (default 1.0)
%     .interferer    - true/false, overlay a second simultaneous LoRa
%                      transmission to model a collision (default false)
%     .interfererSIR_dB - power of desired signal relative to the
%                      colliding interferer, in dB (default 3; LoRa's
%                      "capture effect" typically needs the desired
%                      signal ~6 dB above the interferer to survive)
%
%   Returns the impaired complex baseband receive waveform.

    if nargin < 3, params = struct(); end
    p = setDefault(params, 'SNR_dB', 5);
    p = setDefault(p, 'fadingType', 'rician');
    p = setDefault(p, 'kFactorRician', 3);
    p = setDefault(p, 'maxDopplerHz', 5);
    p = setDefault(p, 'freqOffsetHz', 0);
    p = setDefault(p, 'batteryScale', 1.0);
    p = setDefault(p, 'interferer', false);
    p = setDefault(p, 'interfererSIR_dB', 3);

    sig = txSignal * p.batteryScale;

    % --- Multipath fading representing rubble / NLOS structural debris ---
    switch lower(p.fadingType)
        case 'rician'
            chan = comm.RicianChannel( ...
                'SampleRate', Fs, ...
                'KFactor', 10^(p.kFactorRician/10), ...
                'MaximumDopplerShift', p.maxDopplerHz);
            sig = chan(sig);
        case 'rayleigh'
            chan = comm.RayleighChannel( ...
                'SampleRate', Fs, ...
                'MaximumDopplerShift', p.maxDopplerHz);
            sig = chan(sig);
        case 'none'
            % no fading, clean channel (baseline / best-case sanity check)
        otherwise
            error('Unknown fadingType: %s', p.fadingType);
    end

    % --- Carrier frequency offset (cheap oscillator / thermal drift) ---
    if p.freqOffsetHz ~= 0
        t = (0:numel(sig)-1).' / Fs;
        sig = sig .* exp(1j*2*pi*p.freqOffsetHz*t);
    end

    % --- Collision: overlay a second, independent LoRa-like signal ---
    if p.interferer
        interfererSig = (randn(size(sig)) + 1j*randn(size(sig))) / sqrt(2);
        % scale interferer relative to desired signal power (SIR)
        desiredPower = mean(abs(sig).^2);
        sirLinear = 10^(p.interfererSIR_dB/10);
        interfererPower = desiredPower / sirLinear;
        interfererSig = interfererSig * sqrt(interfererPower / mean(abs(interfererSig).^2));
        % random arrival offset to model asynchronous transmissions
        shift = randi([0, round(0.3*numel(sig))]);
        interfererSig = circshift(interfererSig, shift);
        sig = sig + interfererSig;
        % (SNR_dB below is referenced to the clean full-battery signal,
        % so collision=true adds interference ON TOP of that fixed
        % noise floor rather than redefining what "SNR" means — SNR
        % and SIR now stay independently interpretable.)
    end

    % --- Thermal noise to hit target SNR ---
    % NOTE: we reference the noise power to the ORIGINAL full-battery
    % signal power (txSignal), not the current amplitude of `sig`.
    % awgn(..., 'measured') would instead measure whatever power `sig`
    % happens to have *right now* and add noise to match p.SNR_dB
    % relative to THAT — which silently cancels out batteryScale (and
    % also entangles the interferer overlay into the "SNR" reference).
    % Referencing against the fixed full-power baseline means a
    % depleted battery actually lowers the effective received SNR,
    % which is the whole point of modeling it.
    refPower   = mean(abs(txSignal).^2);
    noisePower = refPower / 10^(p.SNR_dB/10);
    noise = sqrt(noisePower/2) * (randn(size(sig)) + 1j*randn(size(sig)));
    rxSignal = sig + noise;
end

function s = setDefault(s, field, value)
    if ~isfield(s, field) || isempty(s.(field))
        s.(field) = value;
    end
end
