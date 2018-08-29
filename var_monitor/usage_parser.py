'''
Created on Jan 19, 2018

@author: Francesc Torradeflot
'''

import glob
import logging
from datetime import datetime, timedelta
import random
import re 

import matplotlib 
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from itertools import islice,chain
import collections

import marshal
import tempfile
import io
import pstats

import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__file__)

conversion_dict = {'K': -2, 'M': -1, 'G': 0}
def conversion(x):
    """ Converts a given number (x) into a more readable format """
    return x[-1] in conversion_dict and float(x[:-1])*1024.**conversion_dict[x[-1]] or 0.0

def save_or_show(fig, save_plot=False, plot_file=None):
    """ Saves or shows a plot.

        Parameters
        ----------
        fig : IDK________________________________________________
            plot to be saved or shown.
        save_plot : bool
                - True : Plot is saved into a .pdf file.
                - False : Plot is shown but not saved
        plot_file : ________________________________________________
            Name of the file where the plot will be saved.

        Raise
        -----


    """

    print ("SAVE PLOT : ", type(fig))
    if save_plot:
        if plot_file is None:
            raise Exception('File not informed')
        fig.savefig(plot_file)
    else:
        plt.show()

def compute_df_columns(df):
    """ Creates the columns for a given log file storing the data
        in a convenient format

        Parameters
        ----------
        df : IDK________________________________________________
            plot to be saved or shown.
    """

    #print ("COMPUTE DF COLUMNS : ", df, type(df))
    if len(df['timestamp']) == 0:
        return None

    df['timestamp'] = df['timestamp'].apply(datetime.strptime, args=('%Y-%m-%dT%H:%M:%S.%f',))
    df['time_delta_s'] = (df['timestamp'] - df['timestamp'].shift(1)).apply(lambda x: x.total_seconds())
    df['time_spent'] = df['timestamp'] - df['timestamp'][0]
    df['time_spent_s'] = df['time_spent'].apply(lambda x: x.total_seconds())
    total_duration = df['time_spent_s'].iloc[-1]
    if np.isclose(total_duration, 0):
        df['time_spent_rel'] = 0.
    else:
        df['time_spent_rel'] = df['time_spent_s']/df['time_spent_s'].iloc[-1]

    if 'max_vms' in df.columns:
        df['max_vms_GB'] = df['max_vms'].apply(conversion)
    if 'max_rss' in df.columns:
        df['max_rss_GB'] = df['max_rss'].apply(conversion)
    if 'max_uss' in df.columns:
        df['max_uss_GB'] = df['max_uss'].apply(conversion)
    if 'total_io_read' in df.columns:
        df['total_io_read_GB'] = df['total_io_read'].apply(conversion)
    if 'total_io_write' in df.columns:
        df['total_io_write_GB'] = df['total_io_write'].apply(conversion)
    if 'total_cpu_time' in df.columns:
        df['cpu_perc'] = 100.*(df['total_cpu_time'] - df['total_cpu_time'].shift(1))/df['time_delta_s']

    return df

def get_min_2n(some_number):
    '''  find the minimum power of two greater than a given number '''
    return np.power(2., np.ceil(np.log2(some_number)))


def order_by(attr,stats_list):
	for i in range(1,len(stats_list)):
		for j in range(0, len(stats_list)-i):
			if(stats_list[j].var_dict[attr] < stats_list[j+1].var_dict[attr]):
				k = stats_list[j+1]
				stats_list[j+1] = stats_list[j]
				stats_list[j] = k

	return stats_list

#List of variables to be plotted
VARLIST = ['max_vms_GB', 'max_rss_GB',  'total_io_read_GB', 'total_io_write_GB',
           'total_cpu_time', 'cpu_perc']

TIME_LIST = ['tottime','cumtime']

ATTR_LIST = ['num_calls','nonrec_calls','tottime','cumtime']



class Log_Data():
    def __init__ (self, **kwargs): 
        self._parent_folder = None      #Parent folder name -- contains all the folders with log files 
        self._image_size = 62           #Image size -- max number of jobs to be plotted

        for varname, value in kwargs.items():   #Creates a class attribute for each job type specified
            setattr(self, varname, value)

    def get_attr(self,name):
        """ Returns the class attribute named as the given name """
        return object.__getattribute__(self, name)

    def modify_attr(self,name,val): 
        """ Sets the value (val) for a given attribute (name) """
        object.__setattr__(self,name,val)

    def set_image_size(self, sz):
        self._image_size = sz


    def set_parent_folder(self, pth): 
        self._parent_folder = pth

    @property
    def image_size(self):
        return self._image_size
    

    @property
    def parent_folder(self):
        return self._parent_folder



class Stat:
	def __init__(self, key, data):
		self.funct_name = key
		self.var_dict = {}
		for i in range(len(ATTR_LIST)):
			self.var_dict[ATTR_LIST[i]] = data[i]




class UsageParser():

    def __init__(self, lgdr, jobs, mem=4, wr=10):
        self.log_files = None
        self.dfs = None
        self.additional_stats = None

        self.time_files = []
        self.group_data = {}
        self.group_names = {}

        self.log_path = lgdr        #All log files from a folder
        self.data = []              #Contains the data of all the jobs separated by parent folder (list:Log_Data())
        self.selected_jobs = jobs   #job types that will be used to plot/get stats 

        self.memory_limit = mem     
        self.iow_limit = wr


    def load_usage_files(self): 
        """ Loads all the usage files for the selected jobs into self.data using Log_Data class. """
        folders = glob.glob(self.log_path)

        for fld in folders: 
            usg = glob.glob(fld+"/*/usage_*")
            dic = {}
            for j in self.selected_jobs:
                p = re.compile('(.*)'+j)
                
                for u in usg:
                    if re.search(p, u):
                        if j in dic:
                            dic[j].append(u)
                        else : 
                            dic[j] = [u]
            if dic :
                ld = Log_Data(**dic)
                ld.set_parent_folder(fld)

                self.data.append(ld)


    def order_usages(self, log_files, j, set_size=False):
        """ Loads log_files data by job type and orders them according to their identifier
    
            Parameters
            ----------
                log_files:  list 
                            List of log files (usage_*)

                j:  str
                    Name of the job type to get the data from 

                set_size:   bool
                            Used to set image_size

            Return
            ------
                ordered_data:   collections.OrderedDict()
                                Ordered dictionary where keys are the orderd identifier of the jobs and values are its assigned log data 
                size:   int 
                        set_size=True: returns the number of jobs of the given type (j) 
                        set_size=False: returns 0 
        """
        keys = []
        to_be_ordered = {}

        for usg in log_files:
            df = pd.read_csv(usg, engine='python')
            compute_df_columns(df)

            if not df.empty: 
                nums = re.findall(r'\d+', usg.split('/')[-2])
                n = ''
                for i in range(len(nums)): 
                    n += str(nums[i])+'.'
                n = n[:-1]  #le quitamos el ultimo punto 

                to_be_ordered[n] = df
                keys.append(n)

        keys.sort(key=lambda s: list(map(int, s.split('.'))))
        ordered_data = collections.OrderedDict()

        for k in keys: 
            ordered_data[k] = to_be_ordered[k]

        size=0
        if set_size : 
            match = '1.1.'
            for k in keys : 
                if match in k:
                    size +=1

        return ordered_data,size


    def load_data(self, jobs_to_order, set_size_job=False):
        """ Loads all the logs data into self.data. 

            Parameters
            ----------
                jobs_to_order:  list
                                Types of jobs that will need to be ordered by id    

                set_size_job;   str
                                Job that will be used to fix the size of the plots 
        """
        self.load_usage_files()     #Loads ALL the usage files from the parent folder 

        i=0
        for d in self.data:
            i +=1
            for j in self.selected_jobs: 
                if j in jobs_to_order:
                    usg = d.get_attr(j)

                    if len(usg) > 1 :
                        
                        dfs = {}
                        if set_size_job == j : 
                            size = 0
                            dfs,size = self.order_usages(usg,j, True)
                            
                            d.set_image_size(size)
                        else : 
                            dfs,size = self.order_usages(usg,j)
                    else :
                        dfs = []
                        #print (j, ' -- ', usg) 
                        df = pd.read_csv(usg.pop(), engine='python')
                        compute_df_columns(df)

                        if not df.empty:
                            dfs.append(df)
                else : 
                    for usg in d.get_attr(j): 
                        dfs = []

                        df = pd.read_csv(usg, engine='python')
                        compute_df_columns(df)

                        if not df.empty: 
                            dfs.append(df)

                d.modify_attr(j,dfs)

    def split_data(self, d, max_size=62): 
        """ Creates a list containing the indexes of data splitted in chunks  

            Parameters
            ----------
                d: data  
                max_size: int 
                    Maximum size of the data to be sh

            Return
            ------
                chunk:  list
                        List containing a list for each plot that will be made.

        """
        chunk = []
        sample_size = 0
        add = d.image_size
        x = 0

        for j in self.selected_jobs: 
            sample_size += len(d.get_attr(j))
        #print (j, ' -- sample size : ', sample_size)
        if max_size < d.image_size:
            add = max_size

        #print (d.parent_folder ," -- ",  d.image_size) 

        while x+add < sample_size :
            chunk.append([x,x+add-1]) 
            x+=add 

        if x != sample_size: 
            chunk.append([x,sample_size]) 

        return chunk 


    def key_and_value(self, data):
        """ Gets the keys and data from the given Log_Data """

        data_list = []
        key_list = []

        for j in self.selected_jobs: 
            if type(data.get_attr(j)) is collections.OrderedDict :
                data_list = list(chain(data_list, data.get_attr(j).values()))
                key_list = list(chain(key_list, (data.get_attr(j).keys())))
            else : 
                
                
                for d in data.get_attr(j):
                    data_list.append(d)
                    
                    #print (d, j)
                key_list.append(j)

        return data_list, key_list

    def adjust_spines(self, ax,spines):
        """ Used to remove the an axis from a plot """

        for loc, spine in ax.spines.items():
            if loc in spines:
                spine.set_position(('outward',10)) # outward by 10 points
            else:
                spine.set_color('none') # don't draw spine

        # turn off ticks where there is no spine
        if 'left' in spines:
            ax.yaxis.set_ticks_position('left')
        else:
        # no yaxis ticks
            ax.yaxis.set_ticks([])

        if 'bottom' in spines:
            ax.xaxis.set_ticks_position('bottom')
        else:
        # no xaxis ticks
            ax.xaxis.set_ticks([])


    def io_plot(self, fig, gs, data, chunk, io_write_limit=None,var_list=IO_LIST): 
        """ Plots the IO reads and writes, if the data to be plotted (write) passes io_write_limit it will be shown     
            in red, if not it'll be green. 

            Parameters
            ----------
                fig: Figure where the plot will be saved

                gs: matplotlib.gridspec.GridSpec()
                    Set the position of the plot inside the figure (fig)

                data:   Log_Data
                        Data that has to be plotted 

                chunk:  list 
                        Indexes of the data to be plotted (chunk[0] = index of the beginning, chunk[-1]= index of the end)
    
                io_write_limit:     int 
                                    IOw threshold that shouldn't be surpassed by a job
    
                var_list:   list
                            IOw/r variables to be plotted, defined at the beginning of the script. 


        """

        ax_ind = 0
        chunk_size = chunk[-1]-chunk[0]

        if io_write_limit == None : 
            io_write_limit = self.iow_limit

        for var in var_list:
            ax = plt.subplot(gs[2,ax_ind])
            jobs_data, jobs_key = self.key_and_value(data)


            var_max = []
            for v in islice(jobs_data, chunk[0], chunk[1]+1):
                var_max.append(max(v[var]))

            #FALTA PLANNER
            ax_ind +=1 
            plt.xticks(np.arange(0,chunk_size+1), list(jobs_key)[chunk[0]:chunk[-1]+1], rotation='vertical')

            plt.tick_params(axis='x', labelsize=6)

            plt.plot(np.arange(0, len(var_max)),var_max,'o',label=var, markersize=2.4)
            ax.grid(linestyle='dashdot')
            
            plt.title(str(var))
        fig.align_labels()


    def memory_plot(self, fig, gs, data, chunk, mem_limit=None, var_list=MEM_LIST ):
        """ Plots the memory (vms & rss), if rss > mem_limit -> job data will be printed red on the plot, otherwise it will be green. 

            Parameters
            ----------
                fig: Figure where the plot will be saved

                gs: matplotlib.gridspec.GridSpec()
                    Set the position of the plot inside the figure (fig)

                data:   Log_Data
                        Data that has to be plotted 

                chunk:  list 
                        Indexes of the data to be plotted (chunk[0] = index of the beginning, chunk[-1]= index of the end)

                mem_limit:  int 
                            memory threshold that shouldn't be surpassed by a job

                var_list:   list
                            Memory (rss/vms) variables to be plotted, defined at the beginning of the script. 

        """

        ax_ind = 0
        i=0

        if mem_limit == None : 
            mem_limit = self.memory_limit

        chunk_size = chunk[-1]-chunk[0]

        for var in var_list:
            ax = plt.subplot(gs[ax_ind,:])  
            ax.grid(linestyle='dashdot')    

            jobs_data, jobs_key = self.key_and_value(data)

            mn = {}
            serror = []

            for v in islice(jobs_data, chunk[0], chunk[1]+1): 
                #print (type(v))
                mn[i]=np.mean(v[var])
                serror.append(np.std(v[var]))

                bp = ax.scatter(np.zeros_like(v[var]) + i,v[var], s=8)


                if var == 'max_rss_GB' :
                    if max(v[var]) > mem_limit : 
                        bp.set_facecolor('tomato')
                    else:
                        bp.set_facecolor('yellowgreen') 
                else : 
                    bp.set_facecolor('yellowgreen') 

                i+=1

            if ax_ind == 0 : 
                plt.margins(y=0.6)
                self.adjust_spines(ax, ['left'])
            else : 
                self.adjust_spines(ax, ['left','bottom'])


            plt.ylim(ymin=0)
            plt.title(str(var))
            ax.errorbar(mn.keys(),mn.values(),  yerr=serror, color='black', elinewidth=1, alpha=0.7)
            ax_ind +=1
            
        plt.xticks(np.arange(chunk_size+1,chunk_size*2+2), list(jobs_key)[chunk[0]:chunk[-1]+1], rotation='vertical')


    def cpu_perc_plot(self, fig, gs, data, chunk=None): 
        """ Plots the usage in % of the cpu by the jobs on data. Prints the usage on different colors, usg<25% = red , 25%<usg<50% = 
        yellow, 50%<usg<75% dark green, 75%<usg neon green.

            Parameters
            ----------
                fig: Figure where the plot will be saved

                gs: matplotlib.gridspec.GridSpec()
                    Set the position of the plot inside the figure (fig)

                data:   Log_Data
                        Data that has to be plotted 

                chunk:  list 
                        Indexes of the data to be plotted (chunk[0] = index of the beginning, chunk[-1]= index of the end)

        """

        ax = plt.subplot(gs[3,:])   
        mn = {}
        chunk_size = chunk[-1]-chunk[0]
        i=0

        jobs_data, jobs_key = self.key_and_value(data)

        for v in islice(jobs_data, chunk[0], chunk[1]+1):
            mn[i]= np.mean(v['cpu_perc'])
            perc_mn = np.mean(v['cpu_perc'])
            plt.title('cpu_perc')

            plt.xticks(np.arange(0,chunk_size*2+2), jobs_key[chunk[0]:chunk[-1]+1], rotation='vertical')

            bp = ax.scatter(np.zeros_like(v['cpu_perc']) + i,v['cpu_perc'], s=3)
            if perc_mn < 25 : 
                    bp.set_facecolor('tomato')
            elif perc_mn > 25 and perc_mn < 50:
                bp.set_facecolor('yellow') 
            elif perc_mn > 50 and perc_mn < 75 :
                bp.set_facecolor('mediumseagreen') 
            else:
                bp.set_facecolor('lime') 
            
            ax.grid(linestyle='dashdot')
            #ax.errorbar(perc_mn.keys(),perc_mn.values(),  color='black', elinewidth=1, alpha=0.7)
            i+=1
        ax.plot(list(mn.keys()),list(mn.values()),color='black', linewidth=1, alpha=0.7)
    

    def get_mean_and_max(self,job_name,data):
        """ Computes max and mean value for a type of job, prints the results 
            
            Parameters
            ----------
            job_name : str
                String that contains the name of type of job to analize 

            data : dictionary 
                All the data for a folder of the job to be analized
                key :   job identifier 
                value : data values  
        """

        max_time = 0
        max_ram = 0
        max_IOW = 0
        max_cpuperc = 0

        mt = 0  #mean_time
        mr = 0  #mean_rss
        mwr = 0 #mean_ioWrite
        mcpu = 0


        if not isinstance(data, list):
            data_f = data.values()
            
        else : 
            data_f = data

        for d in data_f :           
            mr += np.mean(d['max_rss_GB'])
            mcpu += np.mean(d['cpu_perc'])

            tmp = np.array(d['time_spent_s'])[-1]
            if max_time < np.array(d['time_spent_s'])[-1] : 
                max_time = tmp 
            mt += tmp

            tmp = np.max(d['max_rss_GB'])
            if max_ram < tmp: 
                max_ram = tmp 

            tmp = np.max(d['cpu_perc'])
            if max_cpuperc < tmp: 
                max_cpuperc = tmp 

            tmp = np.array(d['total_io_write_GB'])[-1]
            if max_IOW < tmp : 
                max_IOW = tmp 
            mwr += tmp

        
        mean_time = mt/len(data)
        mean_ram = mr/len(data)
        mean_IOW = mwr/len(data)
        mean_cpuperc = mcpu/len(data)

        mean_time = int(mean_time/60)+(mean_time%60)*0.01
        max_time = int(max_time/60)+(max_time%60)*0.01

        print (job_name, " MEAN TIME (min.secs) : %.2f " % (mean_time), " -- MAX TIME (min.secs) : %.2f"  % max_time)
        print (job_name," MEAN CPU_PERCENT : %.2f" % mean_cpuperc, " -- MAX CPU_PERCENT : %.2f " % max_cpuperc)
        print (job_name," MEAN MEMORY (GB) : %.2f" % mean_ram, " -- MAX MEMORY : %.2f"  % max_ram)
        print (job_name," MEAN IOW (GB) : %.2f" % mean_IOW, " -- MAX IO/W : %.2f"  % max_IOW)

    def get_job_stats(self): 
        """ Gets a summary of the max / mean for the given data
            
            Parameters 
            ----------
            job_type: list 
                All the job types on the folder ([splitter, TU, detector]) 

            data:   
                key : parent folder where stats will be computed. 
                value : dict() || list 
        """
        print ("JOB STATS : ")
        #key == folder_name | val == cvs info from key folder  
        for val in self.data: 
            #Prints MAX/PEAK values for cpu%, time(s), rss(GB) and GB written by all jobs of the same kind  
            print('')
            print ("FOLDER : ", val.parent_folder, " ______ ")
            print('')

            for j in self.selected_jobs :
                print ('--- '+j+' --- ')
                print ('-------------------- ')
                self.get_mean_and_max(j,val.get_attr(j))
                print('')

            print ('___________________________')
        print ('')


    def plot_all_jobs(self, max_size=62):
        """ Plots all the jobs on the folders on self

            Jobs will be classified per folder and plotted on different pdfs, every page of the pdf will contain 2 memory plots (rss/vms)
            2 IO plots (read/write) and a cpu usage (%) plot. 


        """
        path = '/home/cperalta/Desktop/cosasAcabadas/parser/plots/'
        
        for d in self.data: 
            with PdfPages((path+d.parent_folder.split('/')[-2]+'.pdf')) as pdf:
                print ("FOLDER : ", d.parent_folder, " ______ ")
                cs = self.split_data(d, max_size)

                for c in cs : 
                    fig = plt.figure(figsize=(10, 15))  #,tight_layout=True
                    gs = gridspec.GridSpec(4, 2, hspace=0.05, top=0.94)
                    plt.suptitle("--" +str(d.parent_folder), fontsize=12)
                    self.memory_plot(fig, gs, d,c)
                    gs2 = gridspec.GridSpec(4, 2, hspace=0.3)
                    self.io_plot(fig,gs2,d,c)
                    self.cpu_perc_plot(fig,gs2,d,c)
                    pdf.savefig()


    def load_log_files(self, wildcard_list, t_fil, max_len=None):
        """ Loads log files that will be used for plotting.

            Parameters
            ----------
            wildcard_list : IDK________________________________________________
                path and name of the files to be used
            max_len :
                Maximum number of log files that will be used.

        """
        time_files = []

        log_files = []


        for wildcard in wildcard_list:
            #log_files += glob.glob(wildcard)
            log_files.append(wildcard)

        for f in t_fil:
            time_files.append(self.load_time_files(f))

        # When maximum length is fixed, get the first max_len files
        if not max_len is None:
            log_files = log_files[:max_len]
            time_files = time_files[:max_len]

        self.log_files = log_files

        self.time_files = time_files
        self.load_dfs()

    def load_time_files(self,data_file):
        statsfile = tempfile.NamedTemporaryFile()

        s = open(data_file, 'r+b')

        statsfile.file = s
        statsfile.name = data_file

        rep_list = ['tottime', 'cumtime']

        p2 = pstats.Stats(statsfile.name)
        stats = marshal.load(statsfile.file)
        stats_list = []

        for k in stats:
            stats_list.append(Stat(k, stats[k]))

        ordered_stats = {}

        # se tiene que ordenar para cada atributo que se pasa :
        for var in rep_list:
            ordered_stats[var] = order_by(var, stats_list.copy())

        group_names = {}
        group_data = {}

        for key in ordered_stats:
            for i in range(10):
                if key not in group_names:
                    # print (key, " no existe en la iteracion : ", i, " -- ", ordered_stats['tottime'][0].var_dict['tottime'])
                    group_names[key] = [str(ordered_stats[key][i].funct_name)]
                    group_data[key] = [ordered_stats[key][i].var_dict[key]]
                else:
                    # print (key, " EXISTE en la iteracion : ", i)
                    group_names[key].insert(0, str(ordered_stats[key][i].funct_name))
                    group_data[key].insert(0, ordered_stats[key][i].var_dict[key])


        self.group_data = group_data
        self.group_names = group_names

        return ordered_stats

    def load_dfs(self):
        """ Reads log files and creates dfs list containing every log file
            with data stored in a proper format, ready to be plotted.

            example :
                dfs = [ ] FORMATO !

        """
        dfs = []


        for log_file in self.log_files:
            df = pd.read_csv(log_file, engine='python')
            compute_df_columns(df)
            dfs.append(df)

        self.dfs = dfs

        #print ("LOAD_DFS ", self.dfs, type(self.dfs))



    def plot_sample(self, sample_size=1, var_list=VARLIST, save_plot=False, plot_file=None):

        MARGIN = 0.05

        sample_dfs = random.sample(self.dfs, sample_size)


        n_vars = len(var_list)

        fig = plt.figure(figsize=(8*sample_size, 8*n_vars))

        ax_ind = 1
        for var_name in var_list:
            var_min = np.inf
            var_max = -np.inf
            time_max = -np.inf
            var_axes = []
            for sample_df in sample_dfs:
                ax = fig.add_subplot(n_vars, sample_size, ax_ind)
                var_axes.append(ax)
                ax.plot(sample_df['time_spent_s'], sample_df[var_name])
                ax.set_xlabel('Time Spent')
                ax.set_ylabel(var_name)
                var_max = max([var_max, sample_df[var_name].max()])
                var_min = min([var_min, sample_df[var_name].min()])
                time_max = max([time_max, sample_df['time_spent_s'].max()])
                ax_ind += 1

            var_margin = MARGIN*(var_max - var_min)
            var_lim = [var_min - var_margin, var_max + var_margin]
            time_margin = MARGIN*(time_max)
            time_lim = [-time_margin, time_max + time_margin]
            for ax in var_axes:
                ax.set_ylim(var_lim)
                ax.set_xlim(time_lim)

                if var_name == 'max_uss_GB' :
                    xmin = 0
                    xmax = time_max
                    ymin = ymax = 4
                    line = mlines.Line2D([xmin,xmax], [ymin,ymax], color='green')
                    ax.add_line(line)


        save_or_show(fig, save_plot, plot_file)

    def compute_additional_stats(self, var_list = VARLIST, n_bins=100):

        additional_stats = {}

        # compute mean duration
        durations = [df['time_spent_s'].iloc[-1] for df in self.dfs]
        additional_stats['mean_duration'] = np.mean(durations)
        additional_stats['mean_duration_str'] = timedelta(seconds = additional_stats['mean_duration']).__str__()
        additional_stats['max_duration'] = np.max(durations)
        additional_stats['max_duration_str'] = timedelta(seconds = additional_stats['max_duration']).__str__()

        # compute rss histogram
        if 'max_rss_GB' in var_list:
            mean_rss_hist = np.zeros(n_bins)
            rss_count = 0.
            max_rss_GB = max([df['max_rss_GB'].max() for df in self.dfs])
            print('max_rss_GB: {}'.format(max_rss_GB))
            max_rss_2n = get_min_2n(max_rss_GB)
            hist_bins = np.linspace(0., max_rss_2n, n_bins + 1)
            print('max_rss_2n: {}'.format(max_rss_2n))
            for df in self.dfs:
                h = np.histogram(df['max_rss_GB'], bins=hist_bins)
                rss_hist = h[0].astype(float)
                mean_rss_hist += rss_hist
                rss_count += len(df['max_rss_GB'])
            additional_stats['rss_hist'] = mean_rss_hist/rss_count
            additional_stats['rss_hist_bins'] = hist_bins

            # compute max RSS
            additional_stats['max_rss'] = max([df['max_rss_GB'].max() for df in self.dfs])

        self.additional_stats = additional_stats


    def plot_additional_stats(self, save_plot=False, plot_file=None):

        self.compute_additional_stats()
        hist_bins = self.additional_stats['rss_hist_bins']

        hist_bins_centers = np.array((hist_bins[:-1] + hist_bins[1:])/2)
        fig = plt.figure(figsize=(16., 8.))
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(hist_bins_centers, 100*self.additional_stats['rss_hist'], hist_bins_centers[1] - hist_bins_centers[0]);
        ax.set_ylabel('% of sample')
        ax.set_xlabel('GB RSS')

        save_or_show(fig, save_plot, plot_file)

    def plot_value_range(self, var_list=VARLIST, save_plot=False, plot_file=None):

        n_vars = len(var_list)

        fig = plt.figure(figsize=(8, 8*n_vars))

        x = np.linspace(0, 1, 101)
        interp_dfs = {}
        for var_name in var_list:
            interp_dfs[var_name] = pd.DataFrame([], index=x)

        for i_df, df in enumerate(self.dfs):
            for var_name in var_list:
                interp_dfs[var_name][i_df] = np.interp(x, df['time_spent_rel'], df[var_name])

        for i_df, (var_name, interp_df) in enumerate(interp_dfs.items()):
            interp_arr = interp_df.as_matrix()
            ax = fig.add_subplot(n_vars, 1, i_df + 1)
            arr_0 = np.percentile(interp_arr, 0., axis=1)
            arr_25 = np.percentile(interp_arr, 25., axis=1)
            arr_50 = np.percentile(interp_arr, 50., axis=1)
            arr_75 = np.percentile(interp_arr, 75., axis=1)
            arr_100 = np.percentile(interp_arr, 100., axis=1)
            ax.fill_between(x, arr_0, arr_25, color='lightskyblue')
            ax.fill_between(x, arr_25, arr_75, color='steelblue')
            ax.fill_between(x, arr_75, arr_100, color='lightskyblue')
            ax.plot(x, arr_50, color='darkblue')
            ax.set_ylabel(var_name)
            ax.set_xlabel('time %')
            ax.grid(True)
            ax.set_xlim([0., 1.])

        save_or_show(fig, save_plot, plot_file)

    def plot_time(self, sample_size=2, var_list=TIME_LIST):

        ax_ind = 1
        n_vars = len(var_list)
        fig = plt.figure(figsize=(8, 8 * n_vars))


        for file in self.time_files:
            for key in file :
                ax = fig.add_subplot(n_vars, sample_size, ax_ind)
                ax_ind += 1

                for i, name in enumerate(self.group_names[key]):
                    ax.barh(name[-15:], self.group_data[key][i], label=name[0:45])

                legend = ax.legend()
                frame = legend.get_frame()
                frame.set_facecolor('0.90')
                ax.set_xlabel(key)

                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[::-1], labels[::-1], title='Line', loc="lower right")


        save_or_show(fig)

