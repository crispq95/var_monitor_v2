#import multiprocessing 
import psutil 
import os
import sys
import time 
import datetime 

SIZE = 5000000 # 5 Mb


def main(): 
	list_a = []
	list_b = []
	list_c = []

	time.sleep(2)

	print(datetime.datetime.now())
	for i in range(SIZE):
		list_a.append(i)
		list_b.append(2)

	time.sleep(2)

	for i in range(SIZE):
		list_c.append(list_a[i]*list_b[i])


	parent = psutil.Process()
	
	print("PARENT VMS : ", parent.memory_info().vms)
	print("PARENT RSS : ", parent.memory_info().rss)
	print("PARENT USS : ", parent.memory_full_info().uss)

	time.sleep(2)

main () 
