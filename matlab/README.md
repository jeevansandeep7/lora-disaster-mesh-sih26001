# LoRa Disaster-Mesh Simulation — MATLAB Starter Kit

Implements Steps 1–2 of the project: LoRa CSS physical-layer simulation
and disaster-realistic channel modeling. Fully software — no hardware
or SDR required. Requires MATLAB **Communications Toolbox**.

## Files

| File | Purpose |
|---|---|
| `loraChirpSymbol.m` | Generates one LoRa CSS up-/down-chirp symbol from first principles (MATLAB has no built-in LoRa PHY block). |
| `generateLoRaPacket.m` | Assembles a full packet: preamble, sync word, start-of-frame delimiter, and payload chirps from your data bytes. |
| `loraDemodulateSymbol.m` | Standard dechirp + FFT demodulator, recovers the integer symbol value from a received chirp. |
| `disasterChannelModel.m` | Applies disaster-scenario impairments: Rician/Rayleigh multipath (rubble/NLOS), Doppler from mobility, frequency offset, battery-driven power drift, and simulated packet collisions (interferer overlay). |
| `runLoRaDisasterSim.m` | End-to-end driver: builds a packet, sweeps SNR × fading type × collision on/off, demodulates, computes symbol error rate / estimated PDR, plots results, and exports a CSV dataset. |

## Quick start

1. Open `runLoRaDisasterSim.m` in MATLAB and run it.
2. It will print packet info, run the Monte Carlo sweep (takes ~1–2 min
   depending on `nTrialsPerPoint`), show a PDR-vs-SNR plot, and save
   `lora_disaster_link_quality_dataset.csv`.
3. Try changing `SF` (7 → 12) to see the classic LoRa range/robustness
   trade-off in the resulting plot.

## Where this feeds into the rest of the project

The exported CSV (`SNR_dB, Fading, Collision, SymbolErrorRate,
PDR_estimate`) is a starter version of the **per-link quality dataset**
the GNN link-quality predictor (Section IV-D of the paper) will be
trained on. In the full project you'll extend this to generate many
such links simultaneously across a simulated multi-node mesh topology
(varying node distance, density, and mobility) to build the graph
dataset, rather than a single link swept in isolation.

## Known simplifications (documented for your report's limitations section)

- Payload symbols are mapped directly from bytes mod `2^SF`, skipping
  full LoRaWAN whitening/Hamming FEC/interleaving — sufficient for
  PHY-level robustness studies, not spec-conformant frame encoding.
- The interferer is modeled as a random-power overlay rather than a
  second full LoRa waveform; swap in a second `generateLoRaPacket`
  call for a more realistic collision if your evaluation needs it.
- Synchronization (timing/frequency acquisition from the preamble) is
  assumed ideal; the demodulator is given exact symbol boundaries.
  Add preamble-based sync detection if you want to study acquisition
  failure as a distinct error mode.
