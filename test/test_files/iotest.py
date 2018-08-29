import psutil
import os 
import time 
import sys
import subprocess as sp 


SIZE = 10000000 	# 100 000 000 = 100	M 


""" 	Test script to check that the IO monitorization works fine for var_monitor.py
------------------------------------------------------------------------------------------------

	- Input : 	It reads from test/test_file 38MB for each process 
	- Output : 	It writtes a file for each process of 65MB

	If test fails it will raise an error.  
	
"""

def checkFileSizeGB(f): 
	statinfo = os.stat(f)
	size = statinfo.st_size 	#bytes
	#print ("BYTES : ", size)
	size = size/1024/1024		
	#print ('MB : ', size, 'M')		


def processRead(): 
	f = open('data_file', 'r')

	for i in range(SIZE): 
		f.read(4)
	f.close()

def processWrite(): 
	f = open('data_file', 'w')

	for i in range(SIZE): 
		f.write(str(i)) #i = 24 bytes
	f.close()

def multiprocessWrite(pid): 
	filename = str(pid)+'data_file'
	f = open(filename, 'w')

	for i in range(SIZE): 
		f.write(str(i)) #i = 24 bytes
	f.close()

	if os.path.isfile(filename):
		os.remove(filename)

	return filename


def iotest():
	""" 
	Creates a process tree similar to :

			   P
			   |
		|------|-----|
		C1	   C2	 C3   
		|
	|-------|
   C4       C5
	|		|
   C6	    C7

	Read 	-  (304 MB) All proc read 38MB 
	write 	-  (455 MB) All processes but the parent one (P) write 65KB 	
				
	"""

	filename = 'data_file'
	checkFileSizeGB(filename)
	parent = psutil.Process()
	pid2 = None  
	pid3 = None 

	os.fork()
	if parent.pid != os.getpid(): 
		pid2 = os.getpid()

	p = psutil.Process()

	for i in range(2): 
		if (parent.pid ==  os.getpid()):
			os.fork()
		elif(pid2 == os.getpid()):
			os.fork()
			if os.getpid() != pid2:
				pid3 = os.getpid()
			
	if os.getpid() == pid3: 
		os.fork()
	
	p = psutil.Process()
	
	processRead()

	if parent.pid != os.getpid(): 
		multiprocessWrite(p.pid)


	#waits til children terminates >
	if p.children(): 
		psutil.wait_procs(p.children())

	return p.io_counters().read_chars,p.io_counters().write_chars  

def main(): 
	parent = psutil.Process()
	
	readb,writeb = iotest()

	#if parent.pid == os.getpid(): 
	#	f = open('io_results','w')
	#	to_be_written = str(str(readb)+','+str(writeb))
		
	#	f.write(to_be_written)
	#	f.close() 


	time.sleep(2)

main()
