%% runLoRaDisasterSim.m
% End-to-end demo: generate a LoRa packet, pass it through the
% disaster channel model across a sweep of SNR / fading / collision
% conditions, demodulate, and log symbol error rate. The resulting
% table doubles as a starter "link-quality" dataset for training the
% GNN link-quality predictor in the next project stage.

clear; clc;

%% --- PHY configuration ---
SF = 7;                 % spreading factor (try 7 vs 12 to see the
                         % range/robustness trade-off)
BW = 125e3;              % 125 kHz channel
oversample = 8;
Fs = oversample * BW;

%% --- Message payload (simulated disaster alert) ---
payloadStr = 'HELP-NODE07-LAT12.9-LON77.6';
payloadBytes = uint8(payloadStr);

[txSignal, txSymbols] = generateLoRaPacket(payloadBytes, SF, BW, Fs);
fprintf('Generated packet: %d symbols, %d samples, %.2f ms duration\n', ...
    numel(txSymbols), numel(txSignal), 1000*numel(txSignal)/Fs);

%% --- Sweep configuration representing disaster scenario variability ---
snrRange   = -10:2:10;                 % dB
fadingTypes = {'none', 'rician', 'rayleigh'};
collisionOptions = [false, true];
nTrialsPerPoint = 20;                  % Monte Carlo repeats per config

results = table();
row = 0;

for f = 1:numel(fadingTypes)
    for c = 1:numel(collisionOptions)
        for s = 1:numel(snrRange)
            errCount = 0;
            totalSymbols = 0;

            for trial = 1:nTrialsPerPoint
                params = struct();
                params.SNR_dB = snrRange(s);
                params.fadingType = fadingTypes{f};
                params.kFactorRician = 3;
                params.maxDopplerHz = 5;
                params.freqOffsetHz = 50 * randn();   % random small drift
                params.batteryScale = 0.6 + 0.4*rand(); % simulate battery variability
                params.interferer = collisionOptions(c);
                params.interfererSIR_dB = 3;

                rxSignal = disasterChannelModel(txSignal, Fs, params);

                % Demodulate payload symbols only (skip preamble/sync/SFD)
                symLenSamples = round((2^SF)/BW * Fs);
                preambleLen = 8 + 2;      % preamble + sync chirps
                sfdLen = 2.25;
                payloadStart = round((preambleLen + sfdLen) * symLenSamples) + 1;

                decodedSymbols = zeros(1, numel(txSymbols));
                for k = 1:numel(txSymbols)
                    idxStart = payloadStart + (k-1)*symLenSamples;
                    idxEnd   = min(idxStart + symLenSamples - 1, numel(rxSignal));
                    if idxStart > numel(rxSignal), break; end
                    decodedSymbols(k) = loraDemodulateSymbol( ...
                        rxSignal(idxStart:idxEnd), SF, BW, Fs);
                end

                errCount = errCount + sum(decodedSymbols ~= txSymbols);
                totalSymbols = totalSymbols + numel(txSymbols);
            end

            ser = errCount / totalSymbols;               % symbol error rate
            pdrEstimate = (1 - ser)^numel(txSymbols);      % naive packet-level estimate

            row = row + 1;
            results.SF(row) = SF;
            results.SNR_dB(row) = snrRange(s);
            results.Fading(row) = string(fadingTypes{f});
            results.Collision(row) = collisionOptions(c);
            results.SymbolErrorRate(row) = ser;
            results.PDR_estimate(row) = pdrEstimate;
        end
    end
end

disp(results);

%% --- Save dataset for downstream GNN link-quality model training ---
writetable(results, 'lora_disaster_link_quality_dataset.csv');
fprintf('Saved dataset: lora_disaster_link_quality_dataset.csv (%d rows)\n', height(results));

%% --- Quick visualization: PDR vs SNR per fading/collision condition ---
figure;
hold on;
markers = {'-o','-s'};
for f = 1:numel(fadingTypes)
    for c = 1:numel(collisionOptions)
        mask = results.Fading == string(fadingTypes{f}) & results.Collision == collisionOptions(c);
        plot(results.SNR_dB(mask), results.PDR_estimate(mask), markers{c}, ...
            'DisplayName', sprintf('%s, collision=%d', fadingTypes{f}, collisionOptions(c)));
    end
end
xlabel('SNR (dB)'); ylabel('Estimated Packet Delivery Ratio');
title(sprintf('LoRa Link Reliability under Disaster Channel Conditions (SF%d)', SF));
legend('Location', 'southeast'); grid on;
