# Training a Quadruped from Scratch · Isaac Lab Notes

Field notes from training a custom 12-DoF quadruped (ZHAW Wild V1) end to end
in Isaac Lab. URDF design, export pitfalls, env configs, PPO, kick-test for
robustness. Companion code and assets bundled next to the post.

> **Read the post**: open [`index.html`](index.html) in a browser served
> over `http://`. Browsers block `file://` fetches that the embedded 3D
> viewer relies on.

## Run locally

```bash
cd blog
python -m http.server 8000
```

Then open <http://localhost:8000/>.

## GitHub Pages

This folder is self-contained — push it to a repo and enable Pages from the
folder root. Everything the post links to lives next to it: URDF, STLs, all
custom Python files. No `file://` issues, no missing assets.

## Repo layout

```
blog/
├── index.html                       The post + embedded Three.js URDF viewer
├── README.md                          You are here
│
├── quadruped_zhawild_v1/              Robot model
│   ├── quadruped2.urdf                Source of truth — Isaac Lab consumes this
│   └── *.stl                          11 visual meshes referenced by the URDF
│
└── code/
    ├── asset_cfg/
    │   └── zhawild.py                 ArticulationCfg — drops into
    │                                    IsaacLab/source/isaaclab_assets/.../robots/
    ├── zhawild_task/                  Training task package — drops into
    │   ├── __init__.py                  IsaacLab/source/isaaclab_tasks/.../config/zhawild/
    │   ├── flat_env_cfg.py            Scene, rewards, terminations, events
    │   ├── mdp/__init__.py            Custom reward terms re-export
    │   └── agents/
    │       ├── __init__.py
    │       └── rsl_rl_ppo_cfg.py      PPO hyperparameters
    └── scripts/
        ├── load_zhawild.py            Smoke-test loader for the URDF
        └── play_kickable.py           play.py + keyboard kick controller
```

## What's in the post

1. Sim vs. Lab, in plain words
2. Install on Windows
3. First examples to run
4. Designing in Fusion 360
5. Exporting to URDF (using ACDC4Robot)
6. Loading into Isaac Lab + **live 3D URDF viewer**
7. Building the training environment
8. Reward shaping and the crawl trap
9. Running PPO and reading the log
10. Kick test for robustness
11. What I would do differently
12. Appendix · Source files

## Wiring the code into Isaac Lab

```text
IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/zhawild.py
                                                  ← code/asset_cfg/zhawild.py

IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/
        locomotion/velocity/config/zhawild/
                                                  ← code/zhawild_task/  (whole folder)
```

Isaac Lab auto-imports task packages on startup, which is when the
`gym.register(...)` calls run. After that, the task IDs are known to Gymnasium:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-ZHAWild-v0 --headless --num_envs=1024
```

## Author

Zaniyar Jahany · ZHAW · <jaha@zhaw.ch>
