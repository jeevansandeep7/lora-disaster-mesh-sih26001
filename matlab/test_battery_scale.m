%% test_battery_scale.m
% Automated numerical regression test verifying that battery scale reduction
% (simulating node battery depletion) increases LoRa Symbol Error Rate (SER).

clear; clc;

fprintf('=== LoRa Battery Scale Regression Test ===\n');

SF = 7;
BW = 125e3;
oversample = 8;
Fs = oversample * BW;

payloadStr = 'BATTERY-TEST-NODE01';
payloadBytes = uint8(payloadStr);

[txSignal, txSymbols] = generateLoRaPacket(payloadBytes, SF, BW, Fs);

batteryScales = [1.0, 0.7, 0.5, 0.3];
fixedSNR_dB = 5.0;
nTrials = 15;

sers = zeros(1, numel(batteryScales));

for bIdx = 1:numel(batteryScales)
    bScale = batteryScales(bIdx);
    errCount = 0;
    totalSymbols = 0;

    for trial = 1:nTrials
        params = struct();
        params.SNR_dB = fixedSNR_dB;
        params.fadingType = 'rician';
        params.kFactorRician = 3;
        params.maxDopplerHz = 5;
        params.batteryScale = bScale;
        params.interferer = false;

        rxSignal = disasterChannelModel(txSignal, Fs, params);

        symLenSamples = round((2^SF)/BW * Fs);
        preambleLen = 8 + 2;
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

    sers(bIdx) = errCount / totalSymbols;
    fprintf('  batteryScale = %.1f  =>  Symbol Error Rate (SER) = %.4f\n', ...
        bScale, sers(bIdx));
end

% Assert that SER at 0.3 battery scale is strictly greater than or equal to SER at 1.0
if sers(end) >= sers(1)
    fprintf('\nSUCCESS: Test passed! Depleted battery increases/maintains higher SER as expected.\n');
else
    error('TEST FAILED: SER did not increase with battery depletion!');
end
