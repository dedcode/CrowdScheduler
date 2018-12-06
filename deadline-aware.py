from __future__ import print_function, division
import simpy
import numpy
import pandas as pd  # http://pandas.pydata.org/pandas-docs/stable/10min.html
import pickle
import os.path
import time
import gc
import matplotlib
#import copy
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import simpleflock

# todo: cope with the case when a worker will drop and nothing left: even if there are no other jobs the workers should wait for it

options = {}
res = {}
T = time.time()

def main(**kwargs):
    gc.collect()
    global options, res, T  # default values:
    options = {}
    res = {}
    T = time.time()
    res['fairness_index'] = []
    options['log'] = False
    options['filename'] = 'results'
    options['total_time'] = 1800  # in seconds 3600
    options['no_best_effort_batches'] = 1000  #1000
    options['no_workers'] = 1000  # 1000
    options['seed'] = 23  # None
    options['verbose'] = 0
    options['to_stop'] = False
    options['scheduler'] = 'deadline'
    options['average_service_time'] = 11
    options['no_deadline_batches'] = int(100 / 60 * options['total_time'])  # 100 per minute nt(100 / 60 * options['total_time'])
    options['zipf_parameter'] = 2
    options['strictness'] = 0.0  # 0 would be achievable with full parallelisation and 1 with full serial
    for key, value in kwargs.iteritems():
        options[key] = value
        #print(key, value)
    env = simpy.Environment()
    batches = []
    workers = []
    numpy.random.seed(options['seed'])
    arrivals = create_arrivals()
    for i in range(options['no_deadline_batches']):
        batches.append(BatchArrivals(env, i, arrivals['dead_times'][i], arrivals['dead_sizes'][i], arrivals['deadlines'][i]))
    for i in range(options['no_best_effort_batches']):
        index = options['no_deadline_batches'] + i
        batches.append(BatchArrivals(env, index, 0.0, numpy.random.zipf(options['zipf_parameter']), numpy.inf))
    for i in range(options['no_workers']):
        workers.append(WorkerArrivals(env, batches, i, arrivals['work_times'][i]))
    env.run(until=options['total_time']+20)
    res['execution_time'] = time.time() - T
    print('--- RUNNING TIME:', res['execution_time'])
    save(batches)


def create_arrivals():
    dead_times = numpy.cumsum(numpy.random.exponential(options['total_time']/options['no_deadline_batches'],
                                                       options['no_deadline_batches']))
    dead_sizes = numpy.random.zipf(options['zipf_parameter'], options['no_deadline_batches'])
    work_times = numpy.cumsum(numpy.random.exponential(options['total_time'] / options['no_workers'],
                                                       options['no_workers']))
    top_sizes = numpy.argsort(-dead_sizes)[0:int(0.1*options['no_deadline_batches'])]
    #dead_function = lambda x, s: x * s * (1/options['average_service_time'] - 1/(options['average_service_time']*options['no_workers'])) + s/(options['average_service_time']*options['no_workers'])
    dead_function = lambda x, s: s * (x * (options['average_service_time'] - options['average_service_time'] / numpy.min([options['no_workers'], s])) + options['average_service_time'] / numpy.min([options['no_workers'],s]))
    deadlines = numpy.empty(options['no_deadline_batches'])
    deadlines[:] = numpy.inf
    for i in top_sizes:
        deadlines[i] = dead_function(options['strictness'], dead_sizes[i]) + dead_times[i]
    #print(deadlines)
    #print(dead_sizes)
    #print(dead_times)
    return {'dead_times': dead_times, 'dead_sizes': dead_sizes, 'work_times': work_times, 'deadlines': deadlines}


def save(batches):
    deadline_batches_index = numpy.where(numpy.isfinite([x.batch.deadline for x in batches]))
    best_effort_batches_index = numpy.where(numpy.isinf([x.batch.deadline for x in batches]))
    best_effort_completion = [(batches[i].batch.size - batches[i].batch.level)/batches[i].batch.size for i in best_effort_batches_index[0]]
#    print(best_effort_batches_index)
#    print(best_effort_completion)
#    print(numpy.mean(best_effort_completion), numpy.max(best_effort_completion))
    deadline_completion = [(batches[i].batch.size - batches[i].batch.level) / batches[i].batch.size for i in deadline_batches_index[0]]
    ratio_success_deadline = []
    dead_function = lambda x, s: s * (x * (options['average_service_time'] - options['average_service_time'] / numpy.min([options['no_workers'], s])) + options['average_service_time'] / numpy.min([options['no_workers'],s]))
    strict_array = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    fn = options['filename'] + '.pkl'
    with simpleflock.SimpleFlock(options['filename']+'.lock', timeout=60):
        try:
            results = pd.read_pickle(fn)
            print('Results already exists in this directory.. appending...')
        except:
            print('file', fn, "doesn't exists")
            results = pd.DataFrame(
                columns=['total_time', 'no_batches', 'no_workers', 'seed', 'scheduler', 'deadline_success_ratio',
                         'mean_deadline', 'median_deadline', 'std_deadline', 'mean_best_effort', 'median_best_effort',
                         'std_best_effort',
                         'execution_time', 'no_best_effort', 'no_deadline', 'strictness', 'log'])
        for s in range(0, len(strict_array)):
            for i in deadline_batches_index[0]:
                batches[i].batch.deadline = dead_function(strict_array[s], batches[i].batch.size) + batches[i].batch.arrive_time
                batches[i].batch.success = batches[i].batch.completion_time <= batches[i].batch.deadline
            ratio_success_deadline.append(numpy.sum([batches[i].batch.success for i in
                                           deadline_batches_index[0]]) / numpy.size(deadline_batches_index))
            results = results.append(pd.DataFrame(
                {'total_time': options['total_time'], 'no_batches': options['no_best_effort_batches'] +
                                                                    options['no_deadline_batches'],
                 'no_workers': options['no_workers'], 'seed': options['seed'], 'scheduler': options['scheduler'],
                 'execution_time': res['execution_time'], 'deadline_success_ratio': ratio_success_deadline[s],
                         'mean_deadline': numpy.mean(deadline_completion), 'median_deadline': numpy.median(deadline_completion),
                 'std_deadline': numpy.std(deadline_completion), 'mean_best_effort': numpy.mean(best_effort_completion),
                 'median_best_effort': numpy.median(best_effort_completion),
                 'std_best_effort': numpy.std(best_effort_completion), 'no_best_effort': numpy.size(best_effort_batches_index),
                 'no_deadline': numpy.size(deadline_batches_index), 'strictness': strict_array[s], 'log': options['log']}, index=[0]), ignore_index=True)
        results.to_pickle(fn)
        results.to_csv(options['filename']+'.csv', index=False)
    if not numpy.mod(options['seed'], 25):
        plt.plot(*zip(*res['fairness_index']))
        plt.xlabel('Time [s]')
        plt.ylabel('Fairness index')
        axes = plt.gca()
        axes.set_ylim([-3010, 10])
        plt.savefig(options['filename']+'_'+options['scheduler']+'_'+str(options['seed'])+'.eps')
        with open(options['filename'] +str(options['seed'])+options['scheduler']+ '_plot.pkl', 'w') as f:
            pickle.dump(res['fairness_index'], f)


def fairness(x):
    f = -max(numpy.sum(x) / x)
    # b = numpy.argmax(numpy.sum(x)/x)
    return f


def scheduler(env, batches, worker):
    global options
    finished = [x.batch.completed for x in batches]

    def add_point(idx):
        temp = numpy.copy(current_weights)
        temp[idx] = candidates[idx].batch.weight_plus
        f = fairness(temp)
        if numpy.isfinite(f):
            res['fairness_index'].append((env.now, f))
        else:
            res['fairness_index'].append((env.now, -3000.0))

    if all(finished):
        options['to_stop'] = True
        print('--- FINISHED!!!!!!!')
    if options['scheduler'] == 'deadline':
        print('--- SCHEDULER @', env.now, '---') if options['verbose'] > 4 else None
        candidates = [x for x in batches if (x.batch.active) & (x.batch.level > 0)]
        if numpy.size(candidates) == 0:
            print('No jobs available') if options['verbose'] > 4 else None
            return []
        current_weights = [x.batch.weight for x in candidates]
        lowest_deadline = numpy.nanargmin([x.batch.deadline for x in candidates])
        add_point(lowest_deadline)
        print("we got a batch", candidates[lowest_deadline].batch.index) if options['verbose'] > 2 else None
        print("with deadline", candidates[lowest_deadline].batch.deadline) if options['verbose'] > 2 else None
        return candidates[lowest_deadline]
    elif options['scheduler'] == 'fair':
        print('--- SCHEDULER @', env.now, '---') if options['verbose'] > 4 else None
        candidates = [x for x in batches if (x.batch.active) & (x.batch.level > 0)]
        if numpy.size(candidates) == 0:
            print('No jobs available') if options['verbose'] > 4 else None
            return []
        current_weights = [x.batch.weight for x in candidates]
        lowest_level = numpy.nanargmin([x.batch.active_jobs/x.batch.size for x in candidates])
        add_point(lowest_level)
        print("we got a batch", candidates[lowest_level].batch.index) if options['verbose'] > 2 else None
        print("with deadline", candidates[lowest_level].batch.deadline) if options['verbose'] > 2 else None
        return candidates[lowest_level]
    elif options['scheduler'] == 'aware':
        print('--- SCHEDULER @', env.now, '---') if options['verbose'] > 2 else None
        candidates = [x for x in batches if (x.batch.active) & (x.batch.level > 0)]
        if numpy.size(candidates) == 0:
            print('No jobs available') if options['verbose'] > 4 else None
            return []
        #todo: for each candidate batch we have two version of x: chosen and not chosen. Then I have to
        #  compute the index for each choice of one chosen and all the rest not chosen.
        # IMPORTANT: I can compute the current x in the object, and the new x is only adding one piece :)

        #todo choosing beta->inf it means I can just choose the one with smaller
        def best_fairness(xcurrent, xfuture):
            f = numpy.empty(numpy.size(xcurrent))
            for i in range(0, numpy.size(xcurrent)):
                temp = numpy.copy(xcurrent)
                temp[i] = xfuture[i]
                f[i] = fairness(temp)
            return numpy.argmax(f)
        current_weights = numpy.array([x.batch.weight for x in candidates])
        future_weights = numpy.array([x.batch.weight_plus for x in candidates])
        scramble = numpy.random.permutation(range(0, numpy.size(current_weights)))
        fair_batch = scramble[best_fairness(current_weights[scramble], future_weights[scramble])]
        add_point(fair_batch)
        # print("we got a batch", candidates[fair_batch].batch.index) if options['verbose'] > 2 else None
        # print("with deadline", candidates[fair_batch].batch.deadline) if options['verbose'] > 2 else None
        # print("and weight", candidates[fair_batch].batch.weight) if options['verbose'] > 2 else None
        # print('deadlines:')
        # print([x.batch.deadline for x in candidates])
        # print(current_weights)
        # print(fair_batch,'chosen')
        # choice = raw_input('Press enter to continue or q to quit')
        # if choice in ["Q", "q"]: options['to_stop'] = True
        add_point(fair_batch)
        print("we got a batch", candidates[fair_batch].batch.index) if options['verbose'] > 2 else None
        print("with deadline", candidates[fair_batch].batch.deadline) if options['verbose'] > 2 else None
        return candidates[fair_batch]
    elif options['scheduler'] == 'saware':
        print('--- SCHEDULER @', env.now, '---') if options['verbose'] > 2 else None
        candidates = [x for x in batches if (x.batch.active) & (x.batch.level > 0)]
        if numpy.size(candidates) == 0:
            print('No jobs available') if options['verbose'] > 4 else None
            return []
        current_weights = numpy.array([x.batch.weight for x in candidates])
        future_weights = [x.batch.weight_plus for x in candidates]
        scramble = numpy.random.permutation(range(0, numpy.size(current_weights)))
        fair_batch = scramble[numpy.argmin(current_weights[scramble])]
        add_point(fair_batch)
        # print("we got a batch", candidates[fair_batch].batch.index) if options['verbose'] > 2 else None
        # print("with deadline", candidates[fair_batch].batch.deadline) if options['verbose'] > 2 else None
        # print("and weight", candidates[fair_batch].batch.weight) if options['verbose'] > 2 else None
        # print('deadlines:')
        # print([x.batch.deadline for x in candidates])
        # print(current_weights)
        # print(fair_batch,'chosen')
        # choice = raw_input('Press enter to continue or q to quit')
        # if choice in ["Q", "q"]: options['to_stop'] = True
        return candidates[fair_batch]


class BatchArrivals(object):
    def __init__(self, env, index, time_arrival, size, deadline):
        self.batch = simpy.Container(env, init=0, capacity=size)
        self.pp = env.process(self.batch_start(env, index, time_arrival, size, deadline))

    def batch_start(self, env, index, time_arrival, size, deadline):
        self.batch.index = index
        self.batch.active_jobs = 0
        self.batch.completed = False
        self.batch.success = False
        self.batch.active = False  # state of a batch can be nan if still not started or the ratio of completeness
        self.batch.deadline = numpy.nan
        self.show_state(env) if options['verbose'] > 4 else None
        self.batch.arrive_time = time_arrival
        self.batch.process = env.timeout(self.batch.arrive_time)
        self.batch.size = size
        self.batch.completion_time = numpy.inf
        yield self.batch.process
        self.batch.active = True
        self.batch.deadline = deadline  # deadline from starting time
        self.batch.put(self.batch.capacity)
        self.show_state(env) if options['verbose'] > 3 else None
        self.update_weight()

    def update_weight(self):
        if numpy.isinf(self.batch.deadline):  # best effort
            self.batch.weight = self.batch.active_jobs/self.batch.size
            self.batch.weight_plus = (1 + self.batch.active_jobs)/self.batch.size
        else:
            x = 0
            for i in range(0, self.batch.active_jobs):
                if options['log']:
                    x += numpy.log((self.batch.size - self.batch.level - i)/self.batch.size)
                else:
                    x += (self.batch.size - self.batch.level - i) / self.batch.size
            x = x/self.batch.size
            self.batch.weight = x
            if options['log']:
                self.batch.weight_plus = x + numpy.log((self.batch.size - self.batch.level - self.batch.active_jobs)/self.batch.size)
            else:
                self.batch.weight_plus = x + (
                                             self.batch.size - self.batch.level - self.batch.active_jobs) / self.batch.size

    def show_state(self, env):
        print("--- BATCH ---")
        print("Time:", env.now, "Batch ID:", self.batch.index)
        print("Active is", self.batch.active)  # if options['verbose'] > 4 else None
        print("deadline is", self.batch.deadline)  # if options['verbose'] > 4 else None
        print("Number of active jobs", self.batch.active_jobs)
        print("Level", self.batch.level)

    def reserve_job(self):
        self.batch.get(1)
        self.batch.active_jobs += 1
        self.update_weight()

    def success_job(self, env):
        self.batch.active_jobs -= 1
        #print(' --- SUCCESS ', self.batch.level, self.batch.active_jobs)
        if (self.batch.level == 0) & (self.batch.active_jobs == 0):
            print("--- BATCH COMPLETED @", env.now) if options['verbose'] > 2 else None
            print("Batch", self.batch.index, "completed!") if options['verbose'] > 2 else None
            self.batch.active = False
            self.batch.completed = True
            self.batch.completion_time = env.now
            if env.now <= self.batch.deadline:
                print("completed in time! Before", self.batch.deadline) if options['verbose'] > 2 else None
                self.batch.success = True
            else:
                print("not completed in time! After", self.batch.deadline) if options['verbose'] > 2 else None
        self.update_weight()

    def failure_job(self):
        self.batch.active_jobs -= 1
        self.batch.put(1)
        self.update_weight()

class WorkerArrivals(object):
    def __init__(self, env, batches, index, t):
        self.index = index
        self.speed = 1
        self.active = False  # the worker is online (not necessarily working)
        self.working = False
        self.assigned_batch = numpy.nan
        self.arrive_time = t
        self.leave_time = numpy.inf
        #self.show_state(env) if options['verbose'] > 3 else None
        env.process(self.worker_start(env, batches))
        #self.reactivate = env.event()

    def worker_start(self, env, batches):
        yield env.timeout(self.arrive_time)
        self.active = True
        working = env.process(self.worker_manager(env, batches))
        #self.leave_time = numpy.random.uniform(self.arrive_time, options['total_time'])
        stopping = env.timeout(self.leave_time-self.arrive_time)
        self.show_state(env) if options['verbose'] > 3 else None
        yield stopping | working
        if not working.triggered:  # we need to interrupt working if still working when leaving
            working.interrupt("Leaving")
        print('--- WORKER ---') if options['verbose'] > 2 else None
        print('worker', self.index, 'left') if options['verbose'] > 2 else None

    def worker_manager(self, env, batches):
        try:
            while True:
                if options['to_stop']:
                    return
                self.working = True
                while True:
                    if options['to_stop']:
                        return
                    self.assigned_batch = scheduler(env, batches, self)
                    if numpy.size(self.assigned_batch) == 0:
                        batch_events = [x.batch.process for x in batches if x.batch.arrive_time > env.now]
                        if numpy.size(batch_events) == 0:
                            return
                        print('--- WORKER ---') if options['verbose'] > 4 else None
                        print("worker", self.index, 'will resume at', numpy.min([x.batch.arrive_time for x in batches
                            if x.batch.arrive_time > env.now])) if options['verbose'] > 4 else None
                        yield simpy.AnyOf(env, batch_events)
                    else:
                        break
                print('--- WORKER ---') if options['verbose'] > 4 else None
                print("batch", self.assigned_batch, "with deadline", self.assigned_batch.batch.deadline, "assigned") \
                    if options['verbose'] > 4 else None
                self.assigned_batch.reserve_job()
                yield env.timeout(numpy.random.exponential(options['average_service_time']))
                print('--- WORKER ---') if options['verbose'] > 4 else None
                print('worker', self.index, 'finished a job at', env.now,
                      'for batch', self.assigned_batch.batch.index) if options['verbose'] > 4 else None
                self.assigned_batch.success_job(env)
                self.assigned_batch = numpy.nan
        except simpy.Interrupt as i:
            print('--- WORKER ---') if options['verbose'] > 4 else None
            print('job interrupted', env.now, 'because:', i.cause) if options['verbose'] > 2 else None
            if not self.assigned_batch == []:
                self.assigned_batch.failure_job()

    def show_state(self, env):
        print("--- WORKER ---")
        print("Time:", env.now, "worker ID:", self.index)
        print("worker Active?", self.active, "Working?", self.working)  # if options['verbose'] > 4 else None
        print("worker Starts at", self.arrive_time, "Leave at", self.leave_time)  # if options['verbose'] > 4 else None

if __name__ == "__main__":
    main()
