import argparse
from collections import defaultdict
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import tqdm

from dasf.profile.profiler import EventProfiler
from dasf.profile.utils import MultiEventDatabase

class TraceAnalyser:
    def __init__(self, database: MultiEventDatabase, process_trace_before: bool = True):
        self._database = database
        if process_trace_before:
            self._database = list(self._database)
            
    def create_annotated_task_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        
        for event in tqdm.tqdm(self._database, desc="Creating annotated task graph"):
            if event.name == "Compute":
                name = event.args["name"]
                task_key = event.args['key']
                dependencies = event.args['dependencies']
                dependents = event.args['dependents']
                size = event.args['size']
                shape = event.args['shape']
                task_type = event.args['type']

                # Add the task as a node to the graph and store task information as metadata
                graph.add_node(
                    task_key, 
                    name=name, 
                    size=size, 
                    shape=shape, 
                    type=task_type, 
                    duration=event.duration
                )

                # Add the dependencies as edges to the graph
                for dependency in dependencies:
                    graph.add_edge(dependency, task_key)

                # Add the dependents as edges to the graph
                for dependent in dependents:
                    graph.add_edge(task_key, dependent)
        
        for node in graph.nodes:
            input_data_size = sum([graph.nodes[dependency].get('size', 0) for dependency in graph.predecessors(node)])

            # Set the input_data_size attribute for the current node
            graph.nodes[node]['input_data_size'] = input_data_size
            graph.nodes[node]["throughput"] = input_data_size / graph.nodes[node].get("duration", 1)
            
        return graph
    
    def per_function_bottleneck(self):
        # Create the annotated DAG
        graph = self.create_annotated_task_graph()
        
        # Dictionary to store task durations per thread_id
        task_durations = defaultdict(lambda: defaultdict(float))
        # Dictionary to store mean gpu_utilization and gpu_memory_used per task_key
        task_resources = defaultdict(lambda: {'gpu_utilization': [], 'gpu_memory_used': []})
        # Dictionaty mapping name to keys
        task_name_keys = defaultdict(lambda: defaultdict(list))

        # Iterate over the traces to calculate task durations per thread_id
        for event in tqdm.tqdm(self._database, desc="[function_bottleneck] Analysing traces"):
            if event.name == "Compute":
                task_key = event.args['name']
                task_duration = event.duration
                thread_id = event.thread_id
                process_id = event.process_id
                task_name_keys[(process_id, thread_id)][task_key].append(event.args['key'])
                task_durations[(process_id, thread_id)][task_key] += task_duration
            elif event.name == "Resource Usage":
                event_timestamp = event.timestamp
                gpu_utilization = event.args['gpu_utilization']
                gpu_memory_used = event.args['gpu_memory_used']

                # Find the corresponding task for the resource event based on timestamp
                task_key = None
                for task_event in self._database:
                    if task_event.name == "Compute" and task_event.timestamp <= event_timestamp <= (
                            task_event.timestamp + task_event.duration):
                        task_key = task_event.args['name']
                        break

                if task_key is not None:
                    task_resources[task_key]['gpu_utilization'].append(gpu_utilization)
                    task_resources[task_key]['gpu_memory_used'].append(gpu_memory_used)

        # Create a list of dictionaries to store data for the DataFrame
        data = []
        for (process_id, thread_id), durations in tqdm.tqdm(task_durations.items(), desc="[function_bottleneck] Creating dataframe"):
            total_duration = sum(durations.values())
            for task_key, duration in durations.items():
                percentage = (duration / total_duration) * 100
                gpu_utilization_values = task_resources[task_key]['gpu_utilization']
                gpu_memory_used_values = task_resources[task_key]['gpu_memory_used']
                num_tasks = len(task_name_keys[(process_id, thread_id)][task_key])
                mean_data_size = 0
                mean_throughput = 0
                count  = 0
                for name_key in task_name_keys[(process_id, thread_id)][task_key]:
                    mean_data_size += graph.nodes[name_key]["input_data_size"]
                    mean_throughput += graph.nodes[name_key]["throughput"]
                    count += 1
                mean_data_size /= count
                mean_throughput /= count
                
                mean_gpu_utilization = sum(gpu_utilization_values) / len(gpu_utilization_values) if len(
                    gpu_utilization_values) > 0 else 0
                mean_gpu_memory_used = sum(gpu_memory_used_values) / len(gpu_memory_used_values) if len(
                    gpu_memory_used_values) > 0 else 0
                data.append({
                    'Host': process_id, 
                    "GPU": thread_id.split("-")[-1], 
                    'Function': task_key, 
                    'Duration (s)': duration, 
                    'Percentage of total time (%)': percentage,
                    'Mean GPU Utilization (%)': mean_gpu_utilization, 
                    'Mean GPU Memory Used (GB)': mean_gpu_memory_used / 1e9,
                    "Mean Data Size (MB)": mean_data_size / 1e6, 
                    "Mean Throughput (MB/s)": mean_throughput/1e6,
                    "Num Tasks (chunks)": num_tasks,
                    "Mean Task time (s)": duration / num_tasks
                })

        # Create a Pandas DataFrame from the data list
        df = pd.DataFrame(data)
        df.set_index(['Host', 'GPU'], append=True)
        df.sort_values(by='Duration (s)', ascending=False, inplace=True)
        return df
    
    def per_worker_task_balance(self):
        # Dictionary to store the number of tasks per worker at each timestamp
        tasks_per_worker = defaultdict(lambda: defaultdict(int))

        # Find the start and end time
        start_time = float('inf')
        end_time = float('-inf')

        # Iterate over the traces to calculate the number of tasks per worker at each timestamp
        for event in tqdm.tqdm(self._database, desc="[task_balance] Analysing traces"):
            if event.name == "Managed Memory":
                timestamp = int(event.timestamp)
                thread_id = event.thread_id
                tasks = event.args['tasks']
                tasks_per_worker[timestamp][thread_id] = tasks
                
                # Update start and end time
                start_time = min(start_time, timestamp)
                end_time = max(end_time, timestamp)

        # Shift the linear spacing of 1 second in relation to the start time
        timestamps = list(range(0, int(end_time - start_time) + 1))

        # Calculate the mean number of tasks per thread in each time interval
        mean_tasks_per_interval = defaultdict(dict)

        for timestamp in tqdm.tqdm(timestamps, desc="[task_balance] Creating dataframe"):
            shifted_timestamp = start_time + timestamp
            tasks_per_thread = tasks_per_worker[shifted_timestamp]
            for thread_id, tasks in tasks_per_thread.items():
                mean_tasks_per_interval[timestamp][thread_id] = tasks

        # Create a Pandas DataFrame from the mean_tasks_per_interval dictionary
        df = pd.DataFrame.from_dict(mean_tasks_per_interval, orient='index')

        df = df.reindex(sorted(df.columns), axis=1)

        # Fill missing values with 0 (if a thread didn't have any tasks in a specific interval)
        df.fillna(0, inplace=True)

        # Calculate the mean number of tasks per thread across all intervals
        # df['Mean Tasks'] = df.mean(axis=0)

        # Reset the index and rename the column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Time Interval (seconds from begin)'}, inplace=True)
        # df["Time Interval"] = df["Time Interval"].apply(lambda x: x + start_time)

        # Print the DataFrame showing the mean number of tasks per thread in each time interval
        df.sort_values(by='Time Interval (seconds from begin)', inplace=True)
        return df
   
    def per_task_bottleneck(self):
        # Create the annotated DAG
        graph = self.create_annotated_task_graph()
        # Dictionary to store task durations per thread_id
        task_durations = defaultdict(lambda: defaultdict(float))
        # Dictionary to store mean gpu_utilization and gpu_memory_used per task_key
        task_resources = defaultdict(lambda: {'gpu_utilization': [], 'gpu_memory_used': []})
        memory_usage_per_task = defaultdict(int)
        # Dictionaty mapping name to keys
        task_name_keys = defaultdict(lambda: defaultdict(list))

        # Iterate over the traces to calculate task durations per thread_id
        for event in tqdm.tqdm(self._database, desc="[task_bottleneck] Analysing traces"):
            if event.name == "Compute":
                task_key = event.args['key']
                task_duration = event.duration
                thread_id = event.thread_id
                process_id = event.process_id
                task_name_keys[(process_id, thread_id)][task_key].append(event.args['key'])
                task_durations[(process_id, thread_id)][task_key] += task_duration
                memory_usage_per_task[task_key] = event.args['size']

            elif event.name == "Resource Usage":
                event_timestamp = event.timestamp
                gpu_utilization = event.args['gpu_utilization']
                gpu_memory_used = event.args['gpu_memory_used']

                # Find the corresponding task for the resource event based on timestamp
                task_key = None
                for task_event in self._database:
                    if task_event.name == "Compute" and task_event.timestamp <= event_timestamp <= (
                            task_event.timestamp + task_event.duration):
                        task_key = task_event.args['name']
                        break

                if task_key is not None:
                    task_resources[task_key]['gpu_utilization'].append(gpu_utilization)
                    task_resources[task_key]['gpu_memory_used'].append(gpu_memory_used)

        # Create a list of dictionaries to store data for the DataFrame
        data = []
        for (process_id, thread_id), durations in tqdm.tqdm(task_durations.items(), desc="[task_bottleneck] Creating dataframe"):
            total_duration = sum(durations.values())
            for task_key, duration in durations.items():
                percentage = (duration / total_duration) * 100
                gpu_utilization_values = task_resources[task_key]['gpu_utilization']
                gpu_memory_used_values = task_resources[task_key]['gpu_memory_used']
                num_tasks = len(task_name_keys[(process_id, thread_id)][task_key])
                mean_data_size = 0
                mean_throughput = 0
                count  = 0
                for name_key in task_name_keys[(process_id, thread_id)][task_key]:
                    mean_data_size += graph.nodes[name_key]["input_data_size"]
                    mean_throughput += graph.nodes[name_key]["throughput"]
                    count += 1
                mean_data_size /= count
                mean_throughput /= count
                
                mean_gpu_utilization = sum(gpu_utilization_values) / len(gpu_utilization_values) if len(
                    gpu_utilization_values) > 0 else 0
                mean_gpu_memory_used = sum(gpu_memory_used_values) / len(gpu_memory_used_values) if len(
                    gpu_memory_used_values) > 0 else 0
                data.append({
                    'Host': process_id, 
                    "GPU": thread_id.split("-")[-1], 
                    'Task Key': task_key, 
                    'Duration (s)': duration, 
                    'Percentage of total time (%)': percentage,
                    'Memory usage (Mb)': memory_usage_per_task[task_key] / 1e6,
                    # 'Mean GPU Utilization (%)': mean_gpu_utilization, 
                    # 'Mean GPU Memory Used (GB)': mean_gpu_memory_used / 1e9,
                    # "Mean Data Size (MB)": mean_data_size / 1e6, 
                    # "Mean throughput (B/s)": mean_throughput,
                    # "Num Tasks (chunks)": num_tasks,
                })

        # Create a Pandas DataFrame from the data list
        df = pd.DataFrame(data)
        df.set_index(['Host', 'GPU'], append=True)
        df.sort_values(by='Duration (s)', ascending=False, inplace=True)
        return df

valid_analyses = [
    "function_bottleneck",
    "task_bottleneck",
    "task_balance"
]
    
def main(database: MultiEventDatabase, output: str = None, analyses: List[str] = None, head: int = 30):
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)
    
    if analyses is None:
        analyses = valid_analyses
        
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
    
    analyser = TraceAnalyser(database)
    if "function_bottleneck" in analyses:
        df = analyser.per_function_bottleneck()
        if output is not None:
            df.to_csv(f"{output}/function_bottleneck.csv")
            
        print("="*20 + "Function bottleneck" + "="*20)
        print(df.head(head))
        print("=" * 80 + "\n")
        
    if "task_bottleneck" in analyses:
        df = analyser.per_task_bottleneck()
        if output is not None:
            df.to_csv(f"{output}/task_bottleneck.csv")
            
        print("="*20 + "Task bottleneck" + "="*20)
        print(df.head(head))
        print("=" * 80 + "\n")
        
    if "task_balance" in analyses:
        df = analyser.per_worker_task_balance()
        if output is not None:
            df.to_csv(f"{output}/task_balance.csv")
            
        print("="*20 + "Task balance" + "="*20)
        print(df.head(head))
        print("=" * 80 + "\n")
        
    print("Analyses finished!")
    

if __name__ == "__main__":
    # Argument parser with default help format
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--databases", type=str, nargs="+", help="The databases to analyse",required=True)
    parser.add_argument("-o", "--output", type=str, help="The output directory, to save output analysis. If None, print only in screen", required=False)
    parser.add_argument("-a", "--analyses", type=str, nargs="+", help="The analyses to perform (if None, perform all)", required=False)
    
    args = parser.parse_args()
    database = MultiEventDatabase(
        [EventProfiler(database_file=database) for database in args.databases]
    )
    main(database, args.output, args.analyses)