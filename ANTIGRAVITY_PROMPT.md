# Task prompt for Antigravity — complete the LoRa Disaster-Mesh prototype

Paste everything below this line into Antigravity as your task/goal prompt,
with this repository open as the workspace.

---

## Context

This is a hackathon prototype (Smart Amrita Hackathon, problem statement
SIH26001 — "AI-Based early warning and landslide Risk Monitoring System
in NER (Software)"). The pitch: a LoRa mesh network for disaster
scenarios where a GNN predicts per-link quality and an RL agent uses
those predictions to choose routes and spreading factors, beating
non-adaptive baselines (fixed-SF shortest path, and an ADR-like
SNR-threshold heuristic) especially after a disruption event (node
loss).

The repo currently contains a working but deliberately minimal MVP,
built under a 24-48 hour time constraint. It runs end-to-end
(`python/run_demo.py` produces real output), but several pieces are
simplified placeholders that need to be hardened into something you'd
actually defend in front of judges or reviewers. Your job is to
complete it, not redesign it from scratch — keep the existing
architecture and file layout unless you find a specific, justified
reason to change it.

## Repo layout

```
matlab/     LoRa CSS PHY (SF7-SF12) + disaster channel model (Rician/
            Rayleigh fading, collisions, battery decay). Exports
            lora_disaster_link_quality_dataset.csv (Monte Carlo
            SNR/fading/collision -> symbol error rate / PDR).
python/
  lora_mesh/
    mesh.py           random topology gen + log-distance path loss -> SNR
    link_quality.py   interpolates the MATLAB CSV into PDR(snr, fading, collision);
                       falls back to an analytic sigmoid if no CSV is present
    gnn.py            hand-rolled 1-round message-passing net (plain numpy,
                       manual backprop) predicting per-link PDR
    rl_agent.py       tabular Q-learning over (node, next_hop, SF) predicting
                       route + spreading-factor choice
    baselines.py      fixed-SF shortest path (AODV-like) and SNR-threshold
                       adaptive-SF shortest path (ADR-like)
  train_gnn.py        builds a synthetic training set across random topologies,
                       trains gnn.LinkGNN, saves weights (lora_mesh_gnn.npz)
  run_demo.py         end-to-end: build mesh -> train GNN+RL -> inject node-loss
                       disaster event -> let RL re-adapt online -> compare all
                       three policies pre/post-disaster -> save CSV + plot
results/    output of the last run_demo.py run (comparison_results.csv, pdr_comparison.png)
```

Run it with:
```bash
cd python && pip install -r requirements.txt && python run_demo.py
```

## Known simplifications you need to address (in priority order)

1. **The GNN is barely a GNN.** It does exactly one round of
   message passing (concatenating the two endpoint node features with
   the edge features, then an MLP) and is trained with hand-written
   backprop for a 2-layer network. Harden this:
   - Add at least 2 rounds of proper neighbor aggregation (mean or
     attention-weighted over ALL neighbors of a node, not just the
     other endpoint of the edge being scored), so node embeddings
     actually propagate multi-hop context before the edge head reads
     them.
   - Keep it dependency-light if you can (numpy is fine), but if you
     add PyTorch/DGL/PyG, pin exact versions in requirements.txt and
     verify a clean `pip install -r requirements.txt` on a fresh venv
     actually works — don't leave a broken/unpinned dependency.
   - Add a held-out validation split and report validation MSE/MAE in
     train_gnn.py, not just training loss.

2. **The RL agent is tabular and per-topology.** Q-table is keyed by
   raw node IDs, so it does not generalize across different mesh
   instances at all — it only "learns" the one topology it was
   trained on. Either:
   - (preferred, still tractable) re-key state by GNN-predicted link
     quality + hop-distance-to-gateway buckets instead of raw node
     IDs, so the same Q-table generalizes across topologies, or
   - upgrade to a function-approximation policy (small neural net
     taking graph-derived features as state) if time allows.
   Either way, add an evaluation that trains on one random topology
   and tests on several DIFFERENT unseen topologies, and report how
   much performance degrades — right now there is no such
   generalization test and the reported numbers are train=test.

3. **`disasterChannelModel.m` fix needs a regression test.** The bug
   fix (referencing noise power to the full-battery signal instead of
   letting `awgn(...,'measured')` silently cancel out `batteryScale`)
   is only verified by manual code reading, not by a test. Add a
   small MATLAB script/test that sweeps `batteryScale` from 1.0 down
   to 0.3 at fixed SNR_dB and asserts that symbol error rate visibly
   increases as battery depletes — prove the fix actually works
   numerically, not just by inspection.

4. **No real multi-hop MATLAB validation.** The MATLAB side only
   simulates a single point-to-point link. The Python mesh layer
   invents its own SNR-from-distance model independently. At minimum,
   validate that `python/lora_mesh/mesh.py`'s path-loss-derived SNR
   values, when passed through `link_quality.py`, produce PDR curves
   that are directionally consistent with the MATLAB
   `lora_disaster_link_quality_dataset.csv` (i.e., write a script that
   plots both on the same axes and confirm they're not wildly
   contradictory).

5. **Baselines are somewhat charitable/uncharitable by hand-tuned
   constants.** `SF_REQUIRED_SNR_DB` and `SF_AIRTIME_REL` in
   baselines.py and rl_agent.py are rough guesses, not derived from
   the actual LoRa symbol time equations already implemented in
   `matlab/loraChirpSymbol.m` (Ts = 2^SF / BW). Recompute airtime
   constants directly from that formula instead of hand-picked
   multipliers, so the energy/latency comparison is defensible.

6. **Statistical rigor.** `run_demo.py` runs a single seed (42) and
   reports point estimates with no confidence intervals. Add multiple
   random seeds (at least 10 independent mesh topologies x disaster
   events), report mean +/- std or a boxplot for PDR/hops/energy per
   policy, and make sure the headline "GNN+RL wins post-disaster"
   claim actually holds up across seeds and isn't a lucky draw from
   seed 42.

7. **Docs.** Update the top-level README.md's "Known simplifications"
   and "Roadmap" sections to reflect what you actually changed —
   don't leave stale claims about limitations you've since fixed.

## Constraints

- Keep everything runnable with a single `pip install -r
  requirements.txt` — no exotic system dependencies, no GPU
  requirement, no cloud API calls. This needs to run on a judge's
  laptop or CI in under a few minutes.
- Don't break `python run_demo.py` as a one-command demo entry point
  — that's what the demo video and live judging will use.
- Preserve the MATLAB files' role as the physics-grounded data source;
  don't replace them with a pure-Python channel model — the PHY-level
  validation is part of the project's credibility.
- If you change file layout or public function signatures, update
  README.md and this prompt file to match — don't leave stale docs.

## Definition of done

- `cd python && pip install -r requirements.txt && python run_demo.py`
  runs clean on a fresh environment and produces results/ output.
- The GNN does real multi-hop message passing and reports validation
  metrics, not just training loss.
- The RL policy is evaluated on topologies it wasn't trained on, with
  numbers reported for that generalization gap.
- The battery-scale bug fix has an automated numeric check, not just a
  code comment.
- The headline comparison (GNN+RL vs. the two baselines, pre- and
  post-disaster) is reported across multiple seeds with variance, not
  a single run.
- README.md accurately reflects the current state of simplifications
  and limitations after your changes.
