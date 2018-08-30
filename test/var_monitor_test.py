try:
	import ConfigParser
except ImportError:
	import configparser as ConfigParser

import pytest
import psutil 
import os 
import uuid
import shlex
import sys
import pandas as pd

from var_monitor import ProcessTreeMonitor

#FOLDER_PATH = '/home/cperalta/Desktop/cosasAcabadas/var_monitor_test'
FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_IO_TEST = FOLDER_PATH+'/test_files/iotest.py'
PATH_MEM_TEST = FOLDER_PATH+'/test_files/mem_test.py'
#PATH_MEM_TEST = '/nfs/pic.es/user/c/cperalta/python_envs/python_3.5.1/ficherosTest/mem_test.c'


PATH_MEM_RESULTS = FOLDER_PATH+'/logdir/usage_test/memory_usage'
PATH_IO_RESULTS = FOLDER_PATH+'/logdir/usage_test/io_results'

conversion_dict = {'K': 1024, 'M': 1024*1024, 'G': 1024*1024*1024}
def conversion(x):
    """ Converts a given number (x) into a more readable format """
    if x[-1] == 'K': 
    	x = float(x[:-1])*conversion_dict['K']
    elif x[-1] == 'M' :
    	x = float(x[:-1])*conversion_dict['M']

    elif x[-1] == 'G' :
    	x = float(x[:-1])*conversion_dict['G']

    return x

def load_data(path,var_list): 
	f = open(path)

	df = pd.read_csv(path, engine='python')

	data = {} 
	for var in var_list: 
		data[var] = df[var][df.index[-1]]       
	f.close()

	return data 

def load_test_data(path,var_list): 
	monit_kwargs = {}

	logdir = 'logdir/'
	logfile = 'usage_{}.csv'.format(uuid.uuid1().hex)
	whole_logfile = os.path.join(logdir, logfile)
	monit_kwargs['log_file'] = whole_logfile


	print (path)
	cmd= shlex.split("python3 "+path)
	#cmd= shlex.split("./mem")
	proc = psutil.Popen(cmd)

	monit_kwargs['var_list'] = var_list
	monitor = ProcessTreeMonitor(proc, **monit_kwargs)

	monitor.start()

	return load_data(whole_logfile, var_list)


def memory_match (): 
	var_list = ['max_vms','max_rss','max_uss']

	correct_data = load_data(PATH_MEM_RESULTS,var_list) 
	testing_data = load_test_data(PATH_MEM_TEST,var_list)

	return correct_data, testing_data

def test_comp_mem():
	correct_data, testing_data = memory_match()

	assert ( conversion(testing_data['max_vms']) > conversion(testing_data['max_rss']) >= conversion(testing_data['max_uss']))

def test_rss():
	correct_data, testing_data = memory_match()

	correct_rss = conversion(correct_data['max_rss'])
	test_rss = conversion(testing_data['max_rss'])

	max_rang = correct_rss+correct_rss*0.05
	min_rang = correct_rss-correct_rss*0.05

	assert( min_rang < test_rss < max_rang), "Value max_rss out of range."

def test_vms():
	correct_data, testing_data = memory_match()

	correct_vms = conversion(correct_data['max_rss'])
	test_vms = conversion(testing_data['max_rss'])

	max_rang = correct_vms+correct_vms*0.05
	min_rang = correct_vms-correct_vms*0.05

	assert( min_rang < test_vms < max_rang), "Value max_vms out of range."

def test_uss():
	correct_data, testing_data = memory_match()

	correct_uss = conversion(correct_data['max_uss'])
	test_uss = conversion(testing_data['max_uss'])

	max_rang = correct_uss+correct_uss*0.05
	min_rang = correct_uss-correct_uss*0.05

	assert( min_rang < test_uss < max_rang), "Value max_uss out of range."


def io_data(): 
	var_list = ['total_io_read', 'total_io_write']

	f = open(PATH_IO_RESULTS)

	correct_data = f.read().split(',')
	testing_data = load_test_data(PATH_IO_TEST, var_list)

	return correct_data, testing_data


def test_IOr():
	correct_data, testing_data = io_data()

	max_rang = float(correct_data[0])+float(correct_data[0])*0.05
	min_rang = float(correct_data[0])-float(correct_data[0])*0.05

	assert (min_rang < conversion(testing_data['total_io_read']) < max_rang), "Value total_io_read out of range."
	

def test_IOw(): 
	correct_data, testing_data = io_data()

	max_rang = float(correct_data[1])+float(correct_data[1])*0.05
	min_rang = float(correct_data[1])-float(correct_data[1])*0.05

	assert (min_rang < conversion(testing_data['total_io_write']) < max_rang), "Value total_io_write out of range."

def main(): 
	print("PATH :", os.path.dirname(os.path.realpath(__file__)))

main()