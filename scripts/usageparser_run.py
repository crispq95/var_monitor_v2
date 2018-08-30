from var_monitor import usage_parser as up

import configparser as ConfigParser
import argparse
import os


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1', ''):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_main_args():
	parser = argparse.ArgumentParser(description="Order csv data by type of job and turn it into plots or get stats.")

	parser.add_argument("--workdir", "-wdr", required=False, help="Logs folder path [example: /pnfs/pic.es/data/astro/euclid/disk/storage/SC456/workdir_SC456_EXT_KIDS_T1_*/log]")
	parser.add_argument("--config", "-cnf", required=False, help="Configuration file path.")

	parser.add_argument("--jobs", "-j", required=False, help="Job list to be used, should be separated by commas. (example: SimExtDetector_pkg,SimPlanner_pkg,SimTU_pkg")
	parser.add_argument("--size_job", "-sz", required=False, help="Job that will be used to choose the size of a plot.")

	parser.add_argument("--memlim", "-m", required=False, help="Job RAM limit (GB)", type=int)
	parser.add_argument("--writlim", "-w", required=False, help="Job write limit (GB)", type=int)

	parser.add_argument("--stats", "-s", required=False, type=str2bool, nargs='?', default=False, help="Show jobs stats. (yes,true.y.t.1)")
	parser.add_argument("--plot", "-pl", required=False, type=str2bool, nargs='?', default=False, help="Plot jobs stats. (no,false,f,n,0)")

	args = parser.parse_args()
	return args

def main():
	#logdir = "/home/cperalta/Desktop/cpq/workdir_SC456_NIP_FLAT_F1_*"
	#jobs = ['SimNipDetector_pkg', 'SimTU_pkg', 'SimPlanner_pkg']
	memory_limit = None
	iow_limit = None
	jobs = []

	folder = ''
	workdir_path = ''
	set_size_job = []

	args = parse_main_args()
	pth = os.path.dirname(os.path.realpath(__file__)).split('/')

	if args.config: 
		config_path = args.config
	else:
		config_path =  os.path.abspath(os.path.join( os.path.dirname(os.path.realpath(__file__)), os.pardir))+'/conf/parser_conf.cfg'

	conf_file = config_path
	config = ConfigParser.ConfigParser()
	config.read(conf_file)

	#select logdir 
	if args.workdir: 
		folder = str(args.workdir)

	elif config.has_option('paths', 'folder_path') and config.has_option('paths', 'workdir_path') : 
		folder = str(config.get('paths', 'folder_path'))
		workdir_path = str(config.get('paths', 'workdir_path'))

	whole_workdir = folder+workdir_path

	if args.memlim:
		memory_limit = int(args.memlim)		 		
	else : 
		if config.has_option('limits', 'RAM'): 
			memory_limit = int(config.get('limits', 'RAM'))	 		

	if args.writlim:
		iow_limit = int(args.writlim)
	else :
		if config.has_option('limits', 'IOW'):
			iow_limit = int(config.get('limits', 'IOW'))

	if args.jobs:
		jobs = args.jobs.split(',') 
	else : 
		if config.has_option('jobs_info', 'job_names') :
			jobs = config.get('jobs_info', 'job_names').split(',') 

	if args.size_job: 
		set_size_job = args.size_job
	else : 
		if config.has_option('jobs_info', 'size_job'):
			set_size_job = config.get('jobs_info', 'size_job')


	if not args.plot and not args.stats:
		raise Exception('One of --plot or --stats required')
	else: 
		parser = up.UsageParser(whole_workdir,jobs,mem=memory_limit, wr=iow_limit)

		parser.load_data(jobs, set_size_job)


		if args.plot: 
			parser.plot_all_jobs('/'.join(pth[0:-1]))
		if args.stats : 
			parser.get_job_stats()


main()