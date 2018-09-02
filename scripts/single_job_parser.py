from var_monitor import usage_parser as up

import configparser as ConfigParser
import argparse
import os
import glob

VARLIST = ['max_vms_GB', 'max_rss_GB',  'total_io_read_GB', 'total_io_write_GB',
           'total_cpu_time', 'cpu_perc']

def parse_main_args():
	parser = argparse.ArgumentParser(description="Order csv data by type of job and turn it into plots or get stats.")

	parser.add_argument("--jobs", "-j", required=True, help="Path for the usage file(s) to be parsed.")
	parser.add_argument("--options", "-opt", required=True, help="Type of plots : 	1 -- F(t) plots for each job.  \
																				2 -- Histogram. \
																				3 -- Value range plots. \
																				For plotting more than one option separate them by commas.")

	parser.add_argument("--size", "-sz", required=False, help="Number of jobs to be added on the plot")
	parser.add_argument("--path", "-pth", required=False, help="Path where plots will be stored")
	parser.add_argument("--var_list", "-vl", required=False, help="List of variables to be plotted. Availables: ['max_vms_GB', 'max_rss_GB',  'total_io_read_GB', 'total_io_write_GB',\
																												'total_cpu_time', 'cpu_perc']")


	args = parser.parse_args()
	return args



def main(): 
	args = parse_main_args()
	size = 0
	var_list = VARLIST

	jobs = args.jobs.split(',')	
	options = args.options.split(',')

	if args.size: 
		size = args.size
	else : 
		print("ELSE ")
		for j in jobs:
			print(j, len(glob.glob(j))) 
			size += len(glob.glob(j))

	if args.path: 
		plot_path = args.path
	else: 
		plot_path =  os.path.abspath(os.path.join( os.path.dirname(os.path.realpath(__file__)), os.pardir))+'/plots/'


	if args.var_list:
		var_list = args.var_list.split(',')


	print ('VAR LIST :', var_list)
	print(jobs,size)
	print("")
	
	parser = up.UsageParser()
	parser.load_log_files(jobs)


	
	if '1' in options :
		parser.plot_sample(save_plot=True, var_list=var_list, sample_size=size, plot_file=plot_path+'single_job.pdf')
	if '2' in options :
			parser.plot_additional_stats(save_plot=True, plot_file=plot_path+'add_stats.pdf')
	if '3' in options :
		parser.plot_value_range(save_plot=True, var_list=var_list, plot_file=plot_path+'value_rang.pdf')


	return 1 


if __name__ == '__main__':
    main()