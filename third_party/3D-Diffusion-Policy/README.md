Sample code to evaluate a TAX3D checkpoint:

```
bash scripts/eval_policy.sh tax3d dedo_proccloth dzujno18 1 0
```

Sample code to evaluate a DP3 checkpoint:

Sample code to train a DP3 checkpoint:
```
bash scripts/train_policy.sh dp3 dedo_proccloth [MODEL NAME] [SEED] [GPU]
```

Note: I've been using seed=1 instead of seed=0 - there was a bug where using seed=0 led to buggy environment resets in PyBullet. Not sure why this is happening, so I just changed the seed instead.

Note: when running policy evals or generating demos, the pybullet GUI must be enabled for higher quality rollout videos, as the OPENGL renderer is not available during the pybullet DIRECT mode.