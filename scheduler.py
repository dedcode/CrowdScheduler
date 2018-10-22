import collections
import datetime
import functools
import logging
import random
import time 
import sys
import timeit
import numpy as np
import threading
from threading import Thread
from scipy.stats import lognorm
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger('schedule')
logger.setLevel(logging.INFO)

class Scheduler(object):
    def __init__(self, name, res, max_jobs=10, submission = {}):
        logger.info('init new scheduler')
        self.name = name
        self.res = res
        self.jobs = []
        self.max_jobs = max_jobs
        self.submission = submission
        self.running = dict.fromkeys(submission.keys(),0)
        self.lock = threading.Lock()
        
    def has_workers(self):
        # logger.warning('crowd pool size %s', len(self.jobs))
        if len(self.jobs) < self.max_jobs:
            return True
        return False
    
    def run_pending(self):
        """
        Run all jobs that are scheduled to run.
        """
        for job in self.jobs:
            self._run_job(job)

    def _run_job(self, job):
        job.start()
        # task is finished ;)
        try:
            self.jobs.remove(job)
            #del job
        except ValueError:
            pass
        
    def append(self, tag, elapsed, runtime):
        self.lock.acquire()
        try:
            self.res = self.res.append({"Scheme": self.name, "Batch": tag, "Time":elapsed , "Runtime": runtime}, ignore_index=True)
        finally:
            self.lock.release()
            
    def assign_worker(self, tag, dist):
        #Todo: add some probabilty to reject the job !
        job = Job(tag, dist, self)
        self.jobs.append(job)
        return job

class Job(Thread):
    def __init__(self, tag, dist, scheduler=None):
        self.tag = tag
        self.factor = 1000
        self.scheduler = scheduler
        self.sample = dist
        Thread.__init__(self)
        self.shutdown_flag = threading.Event()
        
    def run(self):
        start = timeit.default_timer()
        runtime = self.sample[random.randint(0, len(self.sample))] / self.factor
        scheduler.running[self.tag] +=1
        #print(self.tag, runtime)
        # SLEEP!
        time.sleep(runtime)
        elapsed = timeit.default_timer() - big_bang
        #logger.warning('%s, %s, %s', self.tag,  elapsed * self.factor, delay * self.factor)
        #print(self.tag,  elapsed * self.factor, delay * self.factor)
        self.scheduler.append(self.tag, elapsed * self.factor, runtime * self.factor)
        #once finished, remove myself from the queue
        self.scheduler.jobs.remove(self)
        #decrease the count of running jobs
        self.scheduler.running[self.tag] -= 1
        self.shutdown_flag.set()

def lognorm_params(mode, stddev):
    """
    Given the mode and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    p = np.poly1d([1, -1, 0, 0, -(stddev/mode)**2])
    r = p.roots
    sol = r[(r.imag == 0) & (r.real > 0)].real
    shape = np.sqrt(np.log(sol))
    scale = mode * sol
    return shape, scale

def lognorm_exec_time(mode):
    stddev = mode*0.5
    sigma, scale = lognorm_params(mode, stddev)
    return lognorm.rvs(sigma, 0, scale, size=10000000)

def task_consumer(func, scheduler):
    tag = func(scheduler)
    count, dist, mode = scheduler.submission[tag].values()
    # purge from the list when necessary
    if count == 1:
        scheduler.submission.pop(tag)
    else:
        scheduler.submission[tag]['count'] = count-1
    return tag, dist

def getRandom(scheduler):
    return random.choice(list(scheduler.submission.keys()))

def getSJF(scheduler):
    return sorted(scheduler.submission, key=lambda x: (scheduler.submission[x]['mode']))[0]


def getFIFO(scheduler):
    return list(scheduler.submission.keys())[0]

def getRR(scheduler):
    return random.choice(list(scheduler.submission.keys()))

def getFS(scheduler):
    tmp = sorted(scheduler.running, key=scheduler.running.get)
    for i in range(0, len(scheduler.running)):
        tag = tmp[i]
        if tag in scheduler.submission:
            return tag

strategies = [getSJF, getFS,getRR, getFIFO]
head = True

final = pd.DataFrame(columns=('Scheme','Batch', 'Time', 'Runtime'))

for w in [20, 50, 100, 200, 500]:
    for i in range(1, 100):
        for strategy in strategies:
            # init job stack
            # count is the number of tasks in a batch
            # dist is the dist execution of a task following a lognormal with mode
            # we consider a lognormal distribution of the execution time
            submission ={'B1':{'count':50, 'dist':lognorm_exec_time(75), 'mode':75},
                         'B2':{'count':50, 'dist':lognorm_exec_time(40), 'mode':40},
                         'B3':{'count':200, 'dist':lognorm_exec_time(22), 'mode':22},
                         'B4':{'count':100, 'dist':lognorm_exec_time(11), 'mode':11},
                         'B5':{'count':100, 'dist':lognorm_exec_time(36), 'mode':36}
                        }
            max_num_workers = w
            name = strategy.__name__[3:] + '-'+str(w)+'-'+ str(i)
            scheduler = Scheduler(name, final, max_num_workers, submission)
            print('running: ', name)
            # Main test procedure
            big_bang = timeit.default_timer()
            while scheduler.submission:
                if scheduler.has_workers():
                    # put the name of your selection function
                    task, dist = task_consumer(strategy, scheduler)
                    scheduler.assign_worker(task, dist).start()
                else:
                    pass
            #pbar.close()
            #wait for the last X threads to finish
            for job in scheduler.jobs:
                job.join()
            #time.sleep(5)

final.to_csv('full_run.csv', header=head, index_label='Id', index = False, mode = 'w')
head = False