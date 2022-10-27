import wandb
import pickle
import json
import argparse
import os
from datetime import datetime
import subprocess
import numpy as np

wandb_entity = 'ggm-metrics'

def get_run_dir(args):
	for root, subdir, files in os.walk(args.dir):
		if args.id in root and 'config_save.json' in files:
			return root

	raise Exception('could not find id {} in dir {}'.format(args.id, args.dir))




def get_wandb_run_info(run_dir):
	dirs = os.listdir(run_dir)

	wandb_file_name = 'wandb_info.json'
	if wandb_file_name in dirs:
		with open(os.path.join(run_dir, wandb_file_name), 'r') as f:
			wandb_info = json.load(f)

	else:
		wandb_info = dict(wandb_project=[], wandb_id=[])

	return wandb_info



def create_new_wandb_run(args, run_dir, project):
	def load_config(run_dir):
		with open(os.path.join(run_dir, 'config_save.json'), 'r') as f:
			config = json.load(f)
		return config

	config = load_config(run_dir)
	# run_name = 'CC: {}, lr: {}, prop: {}'.format(config['class_conditioning'], config['lr'], config['message_passing'])
	# run_name = 'Soft. temp: {}, beam search: {}, beam search width: {}, bfs: {}, num_nodes: {}'.format(config['softmax_temperature'], config['beam_search'], config['beam_search_width'], config['use_bfs'], config['num_previous_nodes'])
	# run_name = 'test'
	run = wandb.init(project=project, resume='allow', config=config, reinit=True, entity=wandb_entity)

	def make_wandb_info_file(project, run_dir):
		if os.path.exists(os.path.join(run_dir, 'wandb_info.json')):
			with open(os.path.join(run_dir, 'wandb_info.json'), 'r') as f:
				info = json.load(f)
				info['wandb_project'].append(project); info['wandb_id'].append(wandb.run.id)
		else:
			info = {'wandb_project': [project], 'wandb_id': [wandb.run.id]}
		with open(os.path.join(run_dir, 'wandb_info.json'), 'w') as f:
			json.dump(info, f)

	make_wandb_info_file(project, run_dir)
	return run



def resume_wandb_run(project, wandb_id):
	return wandb.init(project=project, entity=wandb_entity, resume=True, id=wandb_id, reinit=True)



def save_results_to_wandb(args, run_dir):
	with open(os.path.join(run_dir, 'results.h5'), 'rb') as f:
		res = []
		while 1:
			try:
				res += [pickle.load(f)]
			except EOFError:
				break
	if len(res) == 1:
		res = res[0]
		temp = []
		for key, val in res.items():
			temp.append(val)
		res = temp

	max_epoch = len(res)
	for epoch in range(wandb.run.step, max_epoch):
		wandb.log(res[epoch], step=epoch)

	try:
		with open(os.path.join(run_dir, 'memory.h5'), 'rb') as f:
			res = []
			while 1:
				try:
					res += pickle.load(f)
				except EOFError:
					break

		x_values = np.arange(len(res)) * 5 / 3600
		data = [[x, y] for (x, y) in zip(x_values, res)]
		table = wandb.Table(data=data, columns = ["Time (h)", "Mem usage (GB)"])
		wandb.log({"my_custom_plot_id" : wandb.plot.line(table, "Time (h)", "Mem usage (GB)", title="Mem usage (GB) vs. time")})

	except FileNotFoundError:
		pass


if __name__ == '__main__':

	def initialize_wandb_run(args, run_dir, project, run_id):
		if project is None or run_id is None:
			return create_new_wandb_run(args, run_dir, project)
		else:
			# print('resuming')
			api = wandb.Api()
			run = api.run('{}/{}/{}'.format(wandb_entity, project, run_id))
			# with open(os.path.join(run_dir, 'prev_epoch_results.h5'), 'rb') as f:
			# 	last_epoch = pickle.load(f)
			# if len(run.history()) <= last_epoch:
			# 	print('skipping ', run_dir)
			# 	return None
			return resume_wandb_run(project, run_id)

	def sync_by_id(args):
		run_dir = get_run_dir(args)
		print('syncing ', run_dir)
		wandb_info = get_wandb_run_info(run_dir)

		# print(args.wandb_project, wandb_info['wandb_project'], len([proj for proj in wandb_info['wandb_project'] if proj == args.wandb_project]) > 0)
		if args.wandb_project is None:
			assert len(wandb_info['wandb_project']) > 0, 'Run {} hasnt been synced to wandb, pass project'.format(run_dir)
		elif len([proj for proj in wandb_info['wandb_project'] if proj == args.wandb_project]) == 0:
			wandb_info['wandb_project'].append(args.wandb_project); wandb_info['wandb_id'].append(None)
		# print(args)
		# print(zip(wandb_info['wandb_project'], wandb_info['wandb_id']))
		for project, run_id in zip(wandb_info['wandb_project'], wandb_info['wandb_id']):
			# print(project, run_id)
			try:
				run = initialize_wandb_run(args, run_dir, project, run_id)
				if run is None:
					continue
				with run:
					save_results_to_wandb(args, run_dir)
					os.system('cp {} {}'.format(os.path.join(run_dir, 'logfile.log'), wandb.run.dir))
					# for file in os.listdir(run_dir):
						# os.system('cp {} {} -r'.format(os.path.join(run_dir, file), wandb.run.dir))

					if os.path.isdir('{}/models'.format(wandb.run.dir)):
						subprocess.run(['zip', '-r', '-j', '{}/models/models.zip'.format(wandb.run.dir), '{}/models/*.h5'.format(wandb.run.dir)])
						subprocess.run(['rm', '{}/models/*.h5'.format(wandb.run.dir)])

					run_dir = wandb.run.dir

				wandb.join()
				subprocess.run(['rm', run_dir, '-r'])
			except:
				pass

		# os.system('zip -r -j {}/models/models.zip {}/models/*.h5'.format(wandb.run.dir, wandb.run.dir))
		# os.system('rm {}/models/*.h5'.format(wandb.run.dir))

	def sync_by_dir_time(args):
		for root, subdir, files in os.walk(args.dir):
			if 'results.h5' in files:
				run_dir = root
				if datetime.fromtimestamp(os.path.getmtime(run_dir)) < args.last_modified:
					continue
				args.id = str(run_dir).split('-')[-1]
				# try:
				sync_by_id(args)
				# except:
				# 	wandb.join()



	parser = argparse.ArgumentParser(description='DGMG')
	parser.add_argument('--id', default = 'None')
	parser.add_argument('--dir', default = 'experiment_results')
	parser.add_argument('--last_modified', default = 'None')
	parser.add_argument('--wandb_project', type=str)
	args = parser.parse_args()

	# args.wandb_project = 'metrics-' + args.dir

	if args.last_modified != 'None':
		args.last_modified = datetime.strptime(args.last_modified, '%b-%d-%Y')
	else:
		args.last_modified = datetime.strptime('Jan-01-2000', '%b-%d-%Y')

	if args.dir != 'experiment_results':
		args.dir = os.path.join('experiment_results', args.dir)

	if args.id != 'None':
		sync_by_id(args)

	else:
		sync_by_dir_time(args)
