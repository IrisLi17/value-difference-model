<!-- # Model-Ensemble Trust-Region Policy Optimization (ME-TRPO)
[Paper](https://arxiv.org/abs/1802.10592)

ME-TRPO is a deep model-based reinforcement learning algorithm that uses neural networks to model both the dynamics and the policy. The dynamics model maintains uncertainty due to limited data through an ensemble of models. The algorithm alternates among adding transitions to a replay buffer, optimizing the dynamics models given the buffer, and optimizing the policy given the dynamics models in [Dyna's style](https://dl.acm.org/citation.cfm?id=122377). This algorithm significantly helps alleviating the *model bias* problem in model-based RL, when the policy exploits the error in the dynamics model. In many Mujoco domains, we show that it can achieve the same final performance as model-free approaches while using 100x less data. Here we assume that the reward function can be specified.

## Set-up
1) Install [rllab](https://github.com/rll/rllab) and [conda](https://conda.io/docs/user-guide/install/index.html).
2) Create a python environment and install dependencies `conda env create -f tf14.yml`.
   > Activate the environment `source activate tf14`.
3) Put this folder inside `rllab/sandbox/thanard/me-trpo` folder.
4) run `python run_model_based_rl.py trpo -env swimmer`.

## Notes
1) Environments: `swimmer`, `snake`, `half-cheetah`, and `hopper` work reliably and converge quickly (in order of hours). `ant` and `humanoid` takes a couple days on a single GPU and are not as reliable.
2) Algorithms:`trpo` works better than `vpg` which works better than `bptt`.
3) To run `snake`, put `vendor/mujoco_models/snake.xml` under `rllab/vendor/mujoco_models`

## Logging
1) The folder is saved in `data/local/ENVNAME/ENVNAME_DATETIME_0001` when running without ec2(by default).
2) `progress.csv` contains `real_current_validation_cost` which is the negative of the reward so far.
3) `info.log` contains the full logs of data collection, dynamics model optimization, and policy optimization. Note that we are minimizing the proxy cost, `estim_validation_cost`. The true cost is shown as `real_validation_cost`, but unseen to the policy optimizer. -->

# value-difference-model

### Prepare the environment

```bash
git clone https://github.com/IrisLi17/value-difference-model
cd value-difference-model
conda create -n <your_name> python=3.5
conda activate <your_name>
pip install -r requirements.txt
```

### Train the model

```bash
python run_model_based_rl.py trpo -env <env_name>
```

`<env_name>` must be one of `half-cheetah`, `swimmer`, `snake`, `ant`, `humanoid`. 

`half-cheetah`, `swimmer`,  `snake` take hours to converge. `ant` takes ~3 days to converge and suffers from segment fault time to time on my machine.

The logging folder is saved in `data/local/<env_name>/<env_name>_DATETIME_0001` by default.
`progress.csv` contains `real_current_validation_cost` which is the negative of the reward so far.

### Visualize

*ongoing*

```bash
python run_model_based_rl.py trpo -env <env_name> -perform
```

