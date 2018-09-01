from var_monitor import usage_parser as up

import configparser as ConfigParser
import argparse
import os

def parse_main_args():
	parser = argparse.ArgumentParser(description="Order csv data by type of job and turn it into plots or get stats.")

	parser.add_argument("--jobs", "-j", required=False, help="Path for the usage file(s) to be parsed.")

	args = parser.parse_args()
	return args



def main(): 
	args = parse_main_args()

	if  args.jobs : 
		jobs = args.jobs.split(',')	
		print ("jobs : ", jobs)
	else : 
		print("Argument -j is needed.")
		print("")
		return 0


	parser = up.UsageParser()
	parser.load_log_files(jobs)
	parser.plot_sample()




	return 1 


if __name__ == '__main__':
    main()