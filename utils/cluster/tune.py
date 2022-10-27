import turibolt as bolt
import yaml
from sklearn.model_selection import ParameterGrid
import subprocess

# with open('config_imagenet64_augment.yaml', 'r') as fp:
with open('utils/cluster/aws.yaml', 'r') as fp:
    config = yaml.load(fp)

bolt_cfg = bolt.get_current_config()
bolt_cfg['resources'] = config['resources']
if 'cluster_options' in config:
    bolt_cfg['cluster_options'] = config['cluster_options']

default_params = {
        # 'data.log_transform': False,
        # 'epochs': 50,
        # 'evaluation.gen_freq': 5,
        # 'evaluation.nll_freq': 5,
        # 'evaluation.eval_qty': 128,
        'evaluation.eval_b_size': 16,

        'model/structured': 'transformer-encoder',
        # 'model/structured': 'transformer-decoder',
        # 'model/structured': 'transformer',
        'model.structured.mlp.timestep_conditioning': 'sin',
        'model.structured.timestep_conditioning.method': 'add',

        # 'model/structured/conditioning': 'bb_preds',
        'model/structured/conditioning': 'feats',
        'model.structured.conditioning.method': 'seq',
        'model/backbone': 'detectron2',

        'optimizer@optimizer.backbone': 'adamw',
        'optimizer.backbone.lr': 1e-5,
        'optimizer.backbone.weight_decay': 0,

        # 'model.structured.conditioning.concat_gt': True,

        # 'model/backbone': 'gt',
        # 'model.backbone.timestep': 0.025,
        # 'model.backbone.static_noise': True,
        # 'model.backbone.num_timesteps': 4000,
        # 'model.backbone.randomize_order': True,

        'model.predict_class': True,

        # 'model.rescale_timesteps': True,

        'diffusion.loss_type': 'simple_regression',
        'data.randomize_order': 'once',
        # 'diffusion.num_timesteps': 50,
        # 'diffusion.loss_type': 'mse',
        # 'diffusion.model_mean_type_bb': 'start_x',
        # 'diffusion.model_var_type': 'fixed_small'

    }

default_params.update(bolt_cfg['parameters'])
tasks = []
cmd = 'train.py'
for k, v in default_params.items():
    cmd += ' %s=%s' % (k, v)


print(cmd)
cmd = '/coreflow/venv/bin/python ' + cmd
process = subprocess.Popen(cmd.split())
process.wait()
