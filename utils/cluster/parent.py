import turibolt as bolt
import yaml
from sklearn.model_selection import ParameterGrid

# with open('config_imagenet64_augment.yaml', 'r') as fp:
with open('utils/cluster/prem.yaml', 'r') as fp:
    config = yaml.load(fp)
config['permissions'] = bolt.get_current_config()['permissions']

cmd = config['command']

param_grid = [
    {
        # 'optimizer@optimizer.backbone': ['adamw'],
        # 'optimizer.backbone.lr': [1e-5],
        # 'optimizer.backbone.weight_decay': [0],
        # 'batch_size': [1],
        # 'epochs': [150],
        # 'dataset': ['tower'],
        # 'max_num_blocks': [1],
        # 'bb_ordering': ['position'],
        # 'bb_ordering': ['none'],
        # 'bb_ordering': ['size'],
        # 'max_num_blocks': [1, 2, 5, 10, 25],
        # 'data.bb_fmt': ['xyxy'],
        # 'model/structured/timestep_conditioning': ['node'],
        # 'model.structured.timestep_conditioning.type': ['sin'],#, 'l_sin'],
        # 'model.structured.timestep_conditioning.method': ['seq'],#, 'add'],
        # 'model.structured.mlp.timestep_conditioning': ['none'],#, 'cat'],
        # 'model.rescale_timesteps': [True, False],

        # 'data.log_transform': [False],
        # 'data.pad_with': ['noise'],
        # 'data.pad_with': ['gt'],
        # 'model.predict_class': [True],
        # 'model.regress_from': ['gt'],
        # 'model.bg_loss_mult': [0.05],

        # 'data.randomize_order': ['once'],
        # 'model.backbone.randomize_order': ['once'],

        # 'model.use_learned_embeds': [True, False],
        # 'model.use_time_embeds': ['learned_1d', 'learned_2d'],

        # 'diffusion.model_mean_type_bb': ['epsilon'],
        'diffusion.model_mean_type_bb': ['prev_x', 'start_x'],
        # 'diffusion.model_mean_type': ['prev_x', 'start_x', 'epsilon'],
        # 'diffusion.model_var_type': ['learned_range'],
        'diffusion.model_var_type': ['learned_range', 'fixed_small'],
        # 'diffusion.loss_type': ['simple_regression'],
        # 'diffusion.loss_type': ['rescaled_mse', 'kl', 'mse', 'rescaled_kl'],
        # 'diffusion.loss_type': ['mse'],
        # 'diffusion.loss_type': ['kl'],
        'diffusion.loss_type': ['rescaled_mse'],
        # 'diffusion.num_timesteps': [25, 50],
        # 'diffusion.num_timesteps': list(range(2, 26)),
        'diffusion.num_timesteps': [2, 5, 10, 15, 20, 25, 50],
        # 'diffusion.num_timesteps': [100, 175, 250, 500],
        # 'diffusion.num_timesteps': [1000, 4000],

        'model/structured': ['transformer-encoder'],
        # 'model/structured': ['transformer-decoder'],
        # 'model/structured': ['transformer'],

        'model/structured/conditioning': ['bb_preds'],
        # 'model/structured/conditioning': ['both'],
        # 'model.structured.conditioning.method': ['seq'],

        'model/backbone': ['detectron2'],
        'model.backbone.nms_thresh': [0.3],
        'model.structured.conditioning.match_input': [True],
        'model.structured.conditioning.top_k': [100],

        'model.structured.conditioning.concat_method': ['feats'],
        #
        # 'model/backbone': ['gt'],
        # 'model.backbone.static_noise': [True],
        # 'model.backbone.timestep': [0.025],
        # 'model.backbone.num_timesteps': [4000],
        # 'model.backbone.randomize_order': [True],

        # 'evaluation.gen_freq': [5],
        # 'evaluation.nll_freq': [5],

        # 'diffusion.model_mean_type': ['prev_x', 'start_x'],
        # 'diffusion.model_var_type': ['learned_range'],
        # 'diffusion.loss_type': ['rescaled_mse', 'kl'],
        # 'diffusion.num_timesteps': [25, 50],#, 75, 100],
        # 'model/structured': ['transformer-encoder'],
        # 'model/backbone': ['gt'],
        # 'model/structured/conditioning': ['bb_preds'],
        # 'model.backbone.timestep': [0, 0.025, 0.05, 0.1],
        # 'model.backbone.static_noise': [True],
        # 'model.structured.dim_feedforward': [1028],
        # 'model.structured.d_model': [512],
        # 'model.structured.num_layers': [6],
        # 'model.structured.nhead': [8],
        # 'model.structured.mlp.num_layers': [2],
        # 'model.structured.mlp.d_model': [512],
    },

]

# bolt.submit(config)
param_list = list(ParameterGrid(param_grid))
tasks = []
for param in param_list:
    name = ''
    name += ' '.join(['{}:{}'.format(k.split('.')[-1], param[k]) for k in sorted(param)])
    config['name'] = name
    cmd_ = cmd
    for key in param:
        cmd_ += ' %s=%s' % (key, param[key])
    config['command'] = cmd_
    # print(config)
    tasks.append(bolt.submit(config))
