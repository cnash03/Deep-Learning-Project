# Configs

| File | Experiment |
|---|---|
| `e0_overfit.yaml` | Pipeline sanity check (overfit 50 images) |
| `e1_unet_r18_ce.yaml` | Baseline |
| `e2_unet_r50_ce.yaml` | Encoder effect |
| `e3_dlv3p_r50_ce.yaml` | Architecture effect |
| `best_arch.yaml` | **Edit after E2/E3** to the winning architecture |
| `e4`–`e7` | Loss comparison on the best architecture |
| `e8_no_pretrain.yaml` | Pretraining ablation (**set loss to the E4–E7 winner**) |

**E9 (seeds):** rerun the two best configs with `--set seed=1 append_seed_to_name=true`, then `seed=2`.
