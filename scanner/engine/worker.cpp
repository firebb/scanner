/* Copyright 2016 Carnegie Mellon University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "scanner/engine/worker.h"
#include "scanner/engine/evaluate_worker.h"
#include "scanner/engine/kernel_registry.h"
#include "scanner/engine/load_worker.h"
#include "scanner/engine/runtime.h"
#include "scanner/engine/save_worker.h"
#include "scanner/engine/table_meta_cache.h"
#include "scanner/engine/python_kernel.h"
#include "scanner/util/cuda.h"
#include "scanner/util/glog.h"

#include <arpa/inet.h>
#include <grpc/grpc_posix.h>
#include <grpc/support/log.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <sys/socket.h>
#include <omp.h>

// For avcodec_register_all()... should go in software video with global mutex
extern "C" {
#include "libavcodec/avcodec.h"
}

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;

namespace scanner {
namespace internal {

namespace {
inline bool operator==(const MemoryPoolConfig& lhs,
                       const MemoryPoolConfig& rhs) {
  return (lhs.cpu().use_pool() == rhs.cpu().use_pool()) &&
         (lhs.cpu().free_space() == rhs.cpu().free_space()) &&
         (lhs.gpu().use_pool() == rhs.gpu().use_pool()) &&
         (lhs.gpu().free_space() == rhs.gpu().free_space());
}
inline bool operator!=(const MemoryPoolConfig& lhs,
                       const MemoryPoolConfig& rhs) {
  return !(lhs == rhs);
}

i32 wait_for_input(std::vector<EvalQueue> &pre_output_queues,
            std::tuple<std::deque<TaskStream>, EvalWorkEntry> &input) {
  for (i32 pu = 0; pu < pre_output_queues.size(); pu++) {
    EvalQueue &input_work_queue = pre_output_queues[pu];
    if (input_work_queue.try_pop(input)) {
      VLOG(1) << "Scheduler receive input from pu:" << pu;
      return pu;
    }
  }
  return -1;
}

bool has_work_to_do(
    std::deque<Intermediate> &buffer_queue,
    std::vector<std::vector<bool>> &pipeline_status,
    Intermediate &work_togo) {
  i32 buffered_works = buffer_queue.size();
  if (buffered_works == 0) return false;

  for (i32 i = 0; i < buffered_works; i++) {
    Intermediate inter = buffer_queue.front();
    buffer_queue.pop_front();
    i32 pu = inter.pu;
    i32 kg = inter.kg;
    if (!pipeline_status[pu][kg]) {
      work_togo = inter;
      // Restore the task order
      i++;
      while (i < buffered_works) {
        inter = buffer_queue.front();
        buffer_queue.pop_front();
        buffer_queue.push_back(inter);
        i++;
      }
      return true;
    } else {
      buffer_queue.push_back(inter);
    }
  }
  return false;
}

void gen_next_stage_tasks(i32 pu, i32 kg,
    std::vector<std::vector<std::pair<i32, i32>>> &col_mapping,
    std::deque<Intermediate> &buffer_queue,
    EvalWorkEntry &work_entry,
    std::deque<TaskStream> &task_streams,
    SchedulerArgs &sArgs) {
  i32 num_post_col = sArgs.num_post_col;
  i32 num_ops = sArgs.num_ops; 
  Profiler &profiler = sArgs.profiler;
  std::vector<OpStage> &pipeline_stages = sArgs.pipeline_stages;
  std::vector<EvalQueue> &post_input_queues = sArgs.post_input_queues;

  std::set<i32> next_stages;
  if (kg == -1) {
    // kg == -1 indicates input from pre_eval
    for (auto& col : col_mapping) {
      for (auto& map : col) {
        next_stages.insert(std::get<0>(map));
      }
    }
  } else {
    OpStage& stage = pipeline_stages[kg]; 
    for (i32 stage_id: stage.children) {
      next_stages.insert(stage_id);
    }
  }

  // We pass the original columns to the first next stage and copy/ref
  // the rest of columns for the others.
  BatchedColumns& output_columns = work_entry.columns;
  std::vector<std::vector<i64>>& output_row_ids = work_entry.row_ids;
  std::vector<bool> need_copy;
  need_copy.resize(output_columns.size(), false);

  for (i32 stage_id : next_stages) {
  	EvalWorkEntry new_entry;
  	new_entry.table_id = work_entry.table_id;
  	new_entry.job_index = work_entry.job_index;
  	new_entry.task_index = work_entry.task_index;
  	new_entry.needs_configure = work_entry.needs_configure;
  	new_entry.needs_reset = work_entry.needs_reset;
  	new_entry.last_in_io_packet = work_entry.last_in_io_packet;
  	new_entry.last_in_task = work_entry.last_in_task;

    i32 num_input_columns = 0; 
    if (stage_id == -1) {
      // Special handle for post_eval.
      num_input_columns = num_post_col;
    } else {
      // For other stages
      OpStage& stage = pipeline_stages[stage_id];
      num_input_columns = stage.num_inputs();
    }
    new_entry.columns.resize(num_input_columns);
    new_entry.row_ids.resize(num_input_columns);
    new_entry.column_handles.resize(num_input_columns);

    for (i32 col = 0; col < col_mapping.size(); col++) {
      auto& mapping = col_mapping[col];
      bool found = false;
      for (auto& map : mapping) {
        if (std::get<0>(map) == stage_id) {
          i32 to_col = std::get<1>(map);
          DeviceHandle handle = work_entry.column_handles[col];
          //new_entry.columns[to_col]
          if (need_copy[col]) {
            ElementList list = 
              copy_or_ref_elements(profiler, handle,
                                 handle, output_columns[col]);
            new_entry.columns[to_col].insert(new_entry.columns[to_col].end(),
                                             list.begin(),
                                             list.end());
          } else {
            new_entry.columns[to_col].insert(new_entry.columns[to_col].end(),
                                             output_columns[col].begin(),
                                             output_columns[col].end());
            need_copy[col] = true;
          }
          new_entry.row_ids[to_col].insert(new_entry.row_ids[to_col].end(),
                                           output_row_ids[col].begin(),
                                           output_row_ids[col].end());
          new_entry.column_handles[to_col] = handle;
          found = true;
          break;
        }  
      }
      assert(found);
    }

    if (stage_id == -1) {
      VLOG(1) << "Scheduler push result to post eval ";
      if (task_streams.size() > 0) {
        for (i32 i = 0; i < num_ops; i++) {
          task_streams.pop_front();
        }
      }
      post_input_queues[pu].push(
          std::make_tuple(task_streams, new_entry));
    } else {
      Intermediate inter;
      inter.pu = pu;
      inter.kg = stage_id;
      inter.entry = new_entry;
      inter.task_streams = task_streams;

      VLOG(1) << "Scheduler new stage task " << pu << "/" << stage_id;
      buffer_queue.push_back(inter);
    }
  }

  // Clear intermediate
  for (i32 col = 0; col < col_mapping.size(); col++) {
    // check if all outputs are used.
    assert(need_copy[col]);
    output_columns[col].clear();
    output_row_ids.clear();
  }
}

void schedule(SchedulerArgs args) {
  std::deque<i32> free_workers;
  i32 num_eval_threads = args.num_eval_threads;
  i32 num_active_threads = num_eval_threads;
  i32 pipeline_instances_per_node = args.pipeline_instances_per_node;

  i32 num_post_col = args.num_post_col;
  std::vector<EvalQueue> &pre_output_queues = args.pre_output_queues;
  std::vector<EvalQueue> &post_input_queues = args.post_input_queues;
  std::vector<std::vector<std::pair<i32, i32>>> &input_col_mapping =
    args.input_col_mapping;
  IntermediateQueue &result_queue = args.result_queue;
  std::vector<IntermediateQueue> &task_queues = args.task_queues;
  std::vector<OpStage> &pipeline_stages = args.pipeline_stages;
  std::vector<std::vector<bool>> &pipeline_status = args.pipeline_status;
  std::deque<Intermediate> buffer_queue;
  Profiler &profiler = args.profiler;

  for (i32 wid = 0; wid < num_eval_threads; wid++) {
    free_workers.push_back(wid);
  }
  
  VLOG(1) << "firebb scheduler start "; 

  while (true) {
    if (free_workers.size() == 0) {
      Intermediate finished;
      result_queue.pop(finished);
      if (finished.is_last) {
        num_active_threads --;
        if (num_active_threads == 0) {
          VLOG(1) << "firebb scheduler exit "; 
          return; // Terminate scheduler thread
        }
        continue;
      }
      free_workers.push_back(finished.wid);
      VLOG(1) << "Scheduler receive result from worker " << finished.wid;
      i32 pu = finished.pu;
      i32 kg = finished.kg;
      EvalWorkEntry& work_entry = finished.entry;
      auto& task_streams = finished.task_streams;

      //TODO: has_result

      OpStage &stage = pipeline_stages[kg];
      pipeline_status[pu][kg] = false;

      gen_next_stage_tasks(pu, kg, stage.output_mapping,
        buffer_queue, work_entry, task_streams, args); 
    }

    Intermediate work_togo;
    while (!has_work_to_do(buffer_queue, pipeline_status, work_togo)) {
      i32 pu = -1;
      std::tuple<std::deque<TaskStream>, EvalWorkEntry> input;
      if ((pu = wait_for_input(pre_output_queues, input)) == -1) {
        Intermediate finished;
        if (result_queue.try_pop(finished)) {
          if (finished.is_last) {
            num_active_threads --;
            if (num_active_threads == 0) {
              VLOG(1) << "firebb scheduler exit "; 
              return; // Terminate scheduler thread
            }
            continue;
          }
          VLOG(1) << "Scheduler receive result from worker " << finished.wid;
          free_workers.push_back(finished.wid);
          i32 pu = finished.pu;
          i32 kg = finished.kg;
          EvalWorkEntry& work_entry = finished.entry;
          auto& task_streams = finished.task_streams;
          OpStage &stage = pipeline_stages[kg];
          pipeline_status[pu][kg] = false;

          //TODO: has_result
          gen_next_stage_tasks(pu, kg, stage.output_mapping,
            buffer_queue, work_entry, task_streams, args); 

        } else {
          std::this_thread::yield();
        }
      } else {
        gen_next_stage_tasks(pu, -1, input_col_mapping,
          buffer_queue, std::get<1>(input), std::get<0>(input), args); 
      }
    }

    assert(free_workers.size() != 0);
    // Assign work
    i32 wid = free_workers.front();
    VLOG(1) << "Scheduler assign work "<< work_togo.entry.job_index
            << ", " << work_togo.entry.task_index << " pu: " << work_togo.pu
            << " kg: " << work_togo.kg << " to worker " << wid;
    free_workers.pop_front();

    task_queues[wid].push(work_togo);
    pipeline_status[work_togo.pu][work_togo.kg] = true;
  }
}

void worker_thread(IntermediateQueue &task_queue,
    IntermediateQueue &result_queue,
    std::vector<std::vector<EvaluateWorker*>> &pipeline_stages,
    std::vector<std::vector<EvaluateWorkerArgs>> &eval_args,
    Profiler& profiler,
    i32 wid) {

  while (true) {
    Intermediate inter;
    task_queue.pop(inter);
    if (inter.is_last) {
      VLOG(1) << "Worker thread " << wid << " exit";
      Intermediate res;
      res.is_last = true;
      result_queue.push(res);
      return;
    }

    EvalWorkEntry& work_entry = inter.entry;
    auto& task_streams = inter.task_streams;

		i32 pu = inter.pu;
		i32 kg = inter.kg;
		VLOG(2) << "Evaluate (N/KI/G: " << wid << "/" << pu << "/"
            << kg << "): processing job task " << work_entry.job_index
            << ", " << work_entry.task_index;
    auto work_start = now();
    EvaluateWorker &worker = std::ref(*pipeline_stages.at(pu).at(kg)); 

    if (task_streams.size() > 0) {
      // Start of a new task. Tell kernels what outputs they should produce.
      std::vector<TaskStream> streams;
      for (i32 i = 0;
          i < eval_args[pu][kg].arg_group.kernel_factories.size();
          ++i) {
        assert(eval_args[pu][kg].arg_group.task_streams_start + i < 
               task_streams.size());
        streams.push_back(task_streams.at(
              eval_args[pu][kg].arg_group.task_streams_start + i));
      }
      worker.new_task(work_entry.job_index, work_entry.task_index, streams);
    }
    
    auto input_entry = work_entry;
    worker.feed(input_entry, profiler);
    EvalWorkEntry output_entry;
    // TODO: item size is not used in yield now.
    bool result = worker.yield(0, output_entry, profiler);
    (void) result;
    assert(result);

    VLOG(2) << "Evaluate (N/KI/G: " << wid << "/" << pu << "/"
            << kg << "): finished task " << work_entry.job_index
            << ", " << work_entry.task_index << " in " << 
            std::chrono::duration_cast<std::chrono::milliseconds>(now() - work_start).count()
            << "ms";
    result_queue.push(Intermediate{pu, kg, task_streams, output_entry,
        wid, 1});
  }
}

void load_driver(LoadInputQueue& load_work,
                 std::vector<EvalQueue>& initial_eval_work,
                 LoadWorkerArgs args) {
  Profiler& profiler = args.profiler;
  LoadWorker worker(args);
  while (true) {
    auto idle_start = now();

    std::tuple<i32, std::deque<TaskStream>, LoadWorkEntry> entry;
    load_work.pop(entry);
    i32& output_queue_idx = std::get<0>(entry);
    auto& task_streams = std::get<1>(entry);
    LoadWorkEntry& load_work_entry = std::get<2>(entry);

    args.profiler.add_interval("idle", idle_start, now());

    if (load_work_entry.job_index() == -1) {
      break;
    }

    VLOG(2) << "Load (N/PU: " << args.node_id << "/" << args.worker_id
            << "): processing job task (" << load_work_entry.job_index() << ", "
            << load_work_entry.task_index() << ")";

    auto work_start = now();

    auto input_entry = load_work_entry;
    worker.feed(input_entry);

    while (true) {
      EvalWorkEntry output_entry;
      i32 io_packet_size = args.io_packet_size;
      if (worker.yield(io_packet_size, output_entry)) {
        auto& work_entry = output_entry;
        work_entry.first = !task_streams.empty();
        work_entry.last_in_task = worker.done();
        initial_eval_work[output_queue_idx].push(
            std::make_tuple(task_streams, work_entry));
        // We use the task streams being empty to indicate that this is
        // a new task, so clear it here to show that this is from the same task
        task_streams.clear();
      } else {
        break;
      }
    }
    profiler.add_interval("task", work_start, now());
    VLOG(2) << "Load (N/PU: " << args.node_id << "/" << args.worker_id
            << "): finished job task (" << load_work_entry.job_index() << ", "
            << load_work_entry.task_index() << "), pushed to worker "
            << output_queue_idx;
  }
  VLOG(1) << "Load (N/PU: " << args.node_id << "/" << args.worker_id
          << "): thread finished";
}

std::map<int, std::mutex> no_pipelining_locks;
std::map<int, std::condition_variable> no_pipelining_cvars;
std::map<int, bool> no_pipelining_conditions;

void pre_evaluate_driver(EvalQueue& input_work, EvalQueue& output_work,
                         PreEvaluateWorkerArgs args) {
  Profiler& profiler = args.profiler;
  PreEvaluateWorker worker(args);
  // We sort inputs into task work queues to ensure we process them
  // sequentially
  std::map<std::tuple<i32, i32>,
           Queue<std::tuple<std::deque<TaskStream>, EvalWorkEntry>>>
      task_work_queue;
  i32 work_packet_size = args.work_packet_size;

  std::tuple<i32, i32> active_job_task = std::make_tuple(-1, -1);
  VLOG(1) << "pre_eval_driver " << std::this_thread::get_id() <<" starts";
  while (true) {
    auto idle_start = now();

    // If we have no work at all or we do not have work for our current task..
    if (task_work_queue.empty() ||
        (std::get<0>(active_job_task) != -1 &&
         task_work_queue.at(active_job_task).size() <= 0)) {
      std::tuple<std::deque<TaskStream>, EvalWorkEntry> entry;
      input_work.pop(entry);


      auto& task_streams = std::get<0>(entry);
      EvalWorkEntry& work_entry = std::get<1>(entry);
      VLOG(1) << "Pre-evaluate (N/KI: " << args.node_id << "/" << args.worker_id
              << "): got work " << work_entry.job_index << " " << work_entry.task_index;
      if (work_entry.job_index == -1) {
        break;
      }

      VLOG(1) << "Pre-evaluate (N/KI: " << args.node_id << "/" << args.worker_id
              << "): "
              << "received job task " << work_entry.job_index << ", "
              << work_entry.task_index;

      task_work_queue[std::make_tuple(work_entry.job_index,
                                      work_entry.task_index)]
          .push(entry);
    }

    args.profiler.add_interval("idle", idle_start, now());

    if (std::get<0>(active_job_task) == -1) {
      // Choose the next task to work on
      active_job_task = task_work_queue.begin()->first;
    }

    // Wait until we have the next io item for the current task
    if (task_work_queue.at(active_job_task).size() <= 0) {
      std::this_thread::yield();
      continue;
    }

    // Grab next entry for active task
    std::tuple<std::deque<TaskStream>, EvalWorkEntry> entry;
    task_work_queue.at(active_job_task).pop(entry);

    auto& task_streams = std::get<0>(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);

    VLOG(1) << "Pre-evaluate (N/KI: " << args.node_id << "/" << args.worker_id
            << "): "
            << "processing job task " << work_entry.job_index << ", "
            << work_entry.task_index;

    auto work_start = now();

    i32 total_rows = 0;
    for (size_t i = 0; i < work_entry.row_ids.size(); ++i) {
      total_rows = std::max(total_rows, (i32)work_entry.row_ids[i].size());
    }

    bool first = work_entry.first;
    bool last = work_entry.last_in_task;

    auto input_entry = work_entry;
    worker.feed(input_entry, first);
    i32 rows_used = 0;
    while (rows_used < total_rows) {
      EvalWorkEntry output_entry;
      if (!worker.yield(work_packet_size, output_entry)) {
        break;
      }

      if (std::getenv("NO_PIPELINING")) {
        no_pipelining_conditions[args.worker_id] = true;
      }

      if (first) {
        output_work.push(std::make_tuple(task_streams, output_entry));
        first = false;
      } else {
        output_work.push(
            std::make_tuple(std::deque<TaskStream>(), output_entry));
      }

      if (std::getenv("NO_PIPELINING")) {
        std::unique_lock<std::mutex> lk(no_pipelining_locks[args.worker_id]);
        no_pipelining_cvars[args.worker_id].wait(lk, [&] {
          return !no_pipelining_conditions[args.worker_id];
        });
      }
      rows_used += work_packet_size;
    }

    if (last) {
      task_work_queue.erase(active_job_task);
      active_job_task = std::make_tuple(-1, -1);
    }

    profiler.add_interval("task", work_start, now());
  }

  VLOG(1) << "Pre-evaluate (N/PU: " << args.node_id << "/" << args.worker_id
          << "): thread finished ";
}

void evaluate_driver(EvalQueue& input_work, EvalQueue& output_work,
                     EvaluateWorkerArgs args) {
  Profiler& profiler = args.profiler;
  EvaluateWorker worker(args);
  while (true) {
    auto idle_pull_start = now();

    std::tuple<std::deque<TaskStream>, EvalWorkEntry> entry;
    input_work.pop(entry);

    auto& task_streams = std::get<0>(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);

    args.profiler.add_interval("idle_pull", idle_pull_start, now());

    if (work_entry.job_index == -1) {
      break;
    }

    VLOG(2) << "Evaluate (N/KI/G: " << args.node_id << "/" << args.ki << "/"
            << args.kg << "): processing job task " << work_entry.job_index
            << ", " << work_entry.task_index;

    auto work_start = now();

    if (task_streams.size() > 0) {
      // Start of a new task. Tell kernels what outputs they should produce.
      std::vector<TaskStream> streams;
      for (i32 i = 0; i < args.arg_group.kernel_factories.size(); ++i) {
        assert(!task_streams.empty());
        streams.push_back(task_streams.front());
        task_streams.pop_front();
      }
      worker.new_task(work_entry.job_index, work_entry.task_index, streams);
    }

    i32 work_packet_size = 0;
    for (size_t i = 0; i < work_entry.columns.size(); ++i) {
      work_packet_size =
          std::max(work_packet_size, (i32)work_entry.columns[i].size());
    }

    auto input_entry = work_entry;
    worker.feed(input_entry, profiler);
    EvalWorkEntry output_entry;
    bool result = worker.yield(work_packet_size, output_entry, profiler);
    (void)result;
    assert(result);

    profiler.add_interval("task", work_start, now());
    VLOG(2) << "Evaluate (N/KI/G: " << args.node_id << "/" << args.ki << "/"
            << args.kg << "): finished task " << work_entry.job_index
            << ", " << work_entry.task_index << " in " << 
            std::chrono::duration_cast<std::chrono::milliseconds>(now() - work_start).count()
            << "ms";

    auto idle_push_start = now();
    output_work.push(std::make_tuple(task_streams, output_entry));
    args.profiler.add_interval("idle_push", idle_push_start, now());

  }
  VLOG(1) << "Evaluate (N/KI: " << args.node_id << "/" << args.ki
          << "): thread finished";
}

void post_evaluate_driver(EvalQueue& input_work, OutputEvalQueue& output_work,
                          PostEvaluateWorkerArgs args) {
  Profiler& profiler = args.profiler;
  PostEvaluateWorker worker(args);
  while (true) {
    auto idle_start = now();

    std::tuple<std::deque<TaskStream>, EvalWorkEntry> entry;
    input_work.pop(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);

    args.profiler.add_interval("idle", idle_start, now());

    if (work_entry.job_index == -1) {
      break;
    }

    VLOG(2) << "Post-evaluate (N/PU: " << args.node_id << "/" << args.id
            << "): processing task " << work_entry.job_index << ", "
            << work_entry.task_index;

    auto work_start = now();

    auto input_entry = work_entry;
    worker.feed(input_entry);
    EvalWorkEntry output_entry;
    bool result = worker.yield(output_entry);
    profiler.add_interval("task", work_start, now());

    if (result) {
      output_entry.last_in_task = work_entry.last_in_task;
      output_work.push(std::make_tuple(args.id, output_entry));
    }

    if (std::getenv("NO_PIPELINING")) {
      {
          std::unique_lock<std::mutex> lk(no_pipelining_locks[args.id]);
          no_pipelining_conditions[args.id] = false;
      }
      no_pipelining_cvars[args.id].notify_one();
    }
  }

  VLOG(1) << "Post-evaluate (N/PU: " << args.node_id << "/" << args.id
          << "): thread finished ";
}

void save_coordinator(OutputEvalQueue& eval_work,
                      std::vector<SaveInputQueue>& save_work) {
  i32 num_save_workers = save_work.size();
  std::map<std::tuple<i32, i32>, i32> task_to_worker_mapping;
  i32 last_worker_assigned = 0;
  while (true) {
    auto idle_start = now();

    std::tuple<i32, EvalWorkEntry> entry;
    eval_work.pop(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);

    //args.profiler.add_interval("idle", idle_start, now());

    if (work_entry.job_index == -1) {
      break;
    }

    auto job_task_id =
        std::make_tuple(work_entry.job_index, work_entry.task_index);
    if (task_to_worker_mapping.count(job_task_id) == 0) {
      // Assign worker to this task
      task_to_worker_mapping[job_task_id] =
          last_worker_assigned++ % num_save_workers;
    }

    i32 assigned_worker = task_to_worker_mapping.at(job_task_id);
    save_work[assigned_worker].push(entry);

    if (work_entry.last_in_task) {
      task_to_worker_mapping.erase(job_task_id);
    }
  }
}

void save_driver(SaveInputQueue& save_work,
                 SaveOutputQueue& output_work,
                 SaveWorkerArgs args) {
  Profiler& profiler = args.profiler;
  SaveWorker worker(args);

  i32 processed = 0;
  i32 active_job = -1;
  i32 active_task = -1;
  while (true) {
    auto idle_start = now();

    std::tuple<i32, EvalWorkEntry> entry;
    save_work.pop(entry);

    i32 pipeline_instance = std::get<0>(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);

    args.profiler.add_interval("idle", idle_start, now());

    if (work_entry.job_index == -1) {
      break;
    }

    VLOG(2) << "Save (N/KI: " << args.node_id << "/" << args.worker_id
            << "): processing job task (" << work_entry.job_index << ", "
            << work_entry.task_index << ")";

    auto work_start = now();

    if (work_entry.job_index != active_job ||
        work_entry.task_index != active_task) {
      active_job = work_entry.job_index;
      active_task = work_entry.task_index;

      worker.new_task(work_entry.table_id, work_entry.task_index,
                      work_entry.column_types);
      processed = 0;
    }
    processed++;

    auto input_entry = work_entry;
    worker.feed(input_entry);

    VLOG(2) << "Save (N/KI: " << args.node_id << "/" << args.worker_id
            << "): finished task (" << work_entry.job_index << ", "
            << work_entry.task_index << ")";

    args.profiler.add_interval("task", work_start, now());

    if (work_entry.last_in_task) {
      output_work.push(std::make_tuple(pipeline_instance, work_entry.job_index,
                                       work_entry.task_index));
    }
  }

  VLOG(1) << "Save (N/KI: " << args.node_id << "/" << args.worker_id
          << "): thread finished ";
}
}

WorkerImpl::WorkerImpl(DatabaseParameters& db_params,
                       std::string master_address, std::string worker_port)
  : watchdog_awake_(true),
    db_params_(db_params),
    state_(State::INITIALIZING),
    master_address_(master_address),
    worker_port_(worker_port) {
  init_glog("scanner_worker");

  set_database_path(db_params.db_path);

  avcodec_register_all();
#ifdef DEBUG
  // Stop SIG36 from grpc when debugging
  grpc_use_signal(-1);
#endif
  // google::protobuf::io::CodedInputStream::SetTotalBytesLimit(67108864 * 4,
  //                                                            67108864 * 2);

  VLOG(1) << "Create master stub";
  master_ = proto::Master::NewStub(
      grpc::CreateChannel(master_address, grpc::InsecureChannelCredentials()));
  VLOG(1) << "Finish master stub";

  storage_ =
      storehouse::StorageBackend::make_from_config(db_params_.storage_config);

  // Set up Python runtime if any kernels need it
  Py_Initialize();
}

WorkerImpl::~WorkerImpl() {
  State state = state_.get();
  state_.set(State::SHUTTING_DOWN);

  try_unregister();
  trigger_shutdown_.set();

  if (watchdog_thread_.joinable()) {
    watchdog_thread_.join();
  }
  delete storage_;
  if (memory_pool_initialized_) {
    destroy_memory_allocators();
  }
}

grpc::Status WorkerImpl::NewJob(grpc::ServerContext* context,
                                const proto::BulkJobParameters* job_params,
                                proto::Result* job_result) {
  // Ensure that only one job is running at a time and that the worker
  // is in idle mode before transitioning to job start
  State state = state_.get();
  bool ready = false;
  while (!ready) {
    switch (state) {
      case RUNNING_JOB: {
        RESULT_ERROR(job_result, "This worker is already running a job!");
        return grpc::Status::OK;
      }
      case SHUTTING_DOWN: {
        RESULT_ERROR(job_result, "This worker is preparing to shutdown!");
        return grpc::Status::OK;
      }
      case INITIALIZING: {
        state_.wait_for_change(INITIALIZING);
        break;
      }
      case IDLE: {
        if (state_.test_and_set(state, RUNNING_JOB)) {
          ready = true;
          break;
        }
      }
    }
    state = state_.get();
  }

  job_result->set_success(true);
  set_database_path(db_params_.db_path);

  // Setup up table metadata cache for use in other operations
  DatabaseMetadata meta =
      read_database_metadata(storage_, DatabaseMetadata::descriptor_path());
  TableMetaCache table_meta(storage_, meta);

  i32 local_id = job_params->local_id();
  i32 local_total = job_params->local_total();
  // Controls if work should be distributed roundrobin or dynamically
  bool distribute_work_dynamically = true;

  timepoint_t base_time = now();
  const i32 work_packet_size = job_params->work_packet_size();
  const i32 io_packet_size = job_params->io_packet_size() != -1
                                 ? job_params->io_packet_size()
                                 : work_packet_size;
  i32 warmup_size = 0;

  OpRegistry* op_registry = get_op_registry();
  std::vector<proto::Job> jobs(job_params->jobs().begin(),
                               job_params->jobs().end());
  std::vector<proto::Op> ops(job_params->ops().begin(),
                             job_params->ops().end());

  DAGAnalysisInfo analysis_results;
  populate_analysis_info(ops, analysis_results);
  // Need slice input rows to know which slice we are in
  determine_input_rows_to_slices(meta, table_meta, jobs, ops, analysis_results);
  remap_input_op_edges(ops, analysis_results);

  // Read final output columns for use in post-evaluate worker
  // (needed for determining column types)
  std::vector<Column> final_output_columns;
  {
    std::string output_name = jobs.at(0).output_table_name();
    const TableMetadata& table = table_meta.at(output_name);
    final_output_columns = table.columns();
  }
  std::vector<ColumnCompressionOptions> final_compression_options;
  for (auto& opts : job_params->compression()) {
    ColumnCompressionOptions o;
    o.codec = opts.codec();
    for (auto& kv : opts.options()) {
      o.options[kv.first] = kv.second;
    }
    final_compression_options.push_back(o);
  }
  assert(final_output_columns.size() == final_compression_options.size());

  // Setup kernel factories and the kernel configs that will be used
  // to instantiate instances of the op pipeline
  KernelRegistry* kernel_registry = get_kernel_registry();
  std::vector<KernelFactory*> kernel_factories;
  std::vector<KernelConfig> kernel_configs;
  i32 num_cpus = db_params_.num_cpus;
  assert(num_cpus > 0);

  i32 total_gpus = db_params_.gpu_ids.size();
  i32 num_gpus = db_params_.gpu_ids.size() / local_total;
  // Should have at least one gpu if there are gpus
  assert(db_params_.gpu_ids.size() == 0 || num_gpus > 0);
  std::vector<i32> gpu_ids;
  {
    i32 start_idx = num_gpus * local_id;
    for (i32 i = 0; i < num_gpus; ++i) {
      gpu_ids.push_back(db_params_.gpu_ids[(start_idx + i) % total_gpus]);
    }
  }

  // Populate kernel_factories and kernel_configs
  for (size_t i = 0; i < ops.size(); ++i) {
    auto& op = ops.at(i);
    const std::string& name = op.name();
    if (is_builtin_op(name)) {
      kernel_factories.push_back(nullptr);
      kernel_configs.emplace_back();
      continue;
    }
    OpInfo* op_info = op_registry->get_op_info(name);

    DeviceType requested_device_type = op.device_type();
    if (requested_device_type == DeviceType::GPU && num_gpus == 0) {
      RESULT_ERROR(job_result,
                   "Scanner is configured with zero available GPUs but a GPU "
                   "op was requested! Please configure Scanner to have "
                   "at least one GPU using the `gpu_ids` config option.");
      return grpc::Status::OK;
    }

    if (!kernel_registry->has_kernel(name, requested_device_type)) {
      RESULT_ERROR(
          job_result,
          "Requested an instance of op %s with device type %s, but no kernel "
          "exists for that configuration.",
          op.name().c_str(),
          (requested_device_type == DeviceType::CPU ? "CPU" : "GPU"));
      return grpc::Status::OK;
    }

    KernelFactory* kernel_factory =
        kernel_registry->get_kernel(name, requested_device_type);
    kernel_factories.push_back(kernel_factory);

    // Setup kernel config with args from Op DAG
    KernelConfig kernel_config;
    kernel_config.node_id = node_id_;
    kernel_config.args =
        std::vector<u8>(op.kernel_args().begin(), op.kernel_args().end());
    const std::vector<Column>& output_columns = op_info->output_columns();
    for (auto& col : output_columns) {
      kernel_config.output_columns.push_back(col.name());
    }

    // Tell kernel what its inputs are from the Op DAG
    // (for variadic inputs)
    auto& input_columns = op_info->input_columns();
    for (int i = 0; i < op.inputs().size(); ++i) {
      auto input = op.inputs(i);
      kernel_config.input_columns.push_back(input.column());
      if (input_columns.size() == 0) {
        // We ccan have 0 columns in op info if variadic arguments
        kernel_config.input_column_types.push_back(ColumnType::Other);
      } else {
        kernel_config.input_column_types.push_back(input_columns[i].type());
      }
    }
    kernel_configs.push_back(kernel_config);
  }

  // Setup Scheduler and evaluate workers
  Profiler scheduler_profiler(base_time);
  i32 pipeline_instances_per_node = job_params->pipeline_instances_per_node();
  i32 num_eval_threads = job_params->num_eval_threads();
  VLOG(1) << "Num worker threads: " << num_eval_threads;

  std::vector<EvalQueue> post_input_queues(pipeline_instances_per_node);
  std::vector<EvalQueue> pre_output_queues(pipeline_instances_per_node);
  IntermediateQueue result_queue;
  std::vector<IntermediateQueue> task_queues(num_eval_threads);
  std::vector<OpStage> pipeline_stages;
  std::vector<std::vector<bool>> pipeline_status(
      pipeline_instances_per_node);
  std::vector<std::vector<std::pair<i32, i32>>> input_col_mapping;

  SchedulerArgs sArgs(pre_output_queues,
                      post_input_queues,
                      input_col_mapping,
                      result_queue,
                      task_queues,
                      pipeline_stages,
                      pipeline_status,
                      scheduler_profiler);
  sArgs.num_eval_threads = num_eval_threads;
  sArgs.pipeline_instances_per_node = pipeline_instances_per_node;
  sArgs.num_post_col = ops.back().inputs().size();

  // compute kernel group size first
  std::map<i32, i32> op_stage_mapping;
  if (!kernel_factories.empty()) {
    i32 kg = 0;
    for (size_t i = 1; i < kernel_factories.size() - 1; ++i) {
      if (ops.at(i).name() == INPUT_OP_NAME)
        continue;
      op_stage_mapping[i] = kg;
      pipeline_stages.emplace_back(kg++);
      pipeline_stages.back().add_op(i);
    }
  }

  // Analyze op DAG to determine what inputs need to be pipped along
  // and when intermediates can be retired -- essentially liveness analysis
  perform_liveness_analysis(ops,
      analysis_results,
      pipeline_stages,
      op_stage_mapping,
      input_col_mapping);
  // The live columns at each op index
  std::vector<std::vector<std::tuple<i32, std::string>>>& live_columns =
      analysis_results.live_columns;
  // The columns to remove for the current kernel
  std::vector<std::vector<i32>> dead_columns =
      analysis_results.dead_columns;
  // Outputs from the current kernel that are not used
  std::vector<std::vector<i32>> unused_outputs =
      analysis_results.unused_outputs;
  // Indices in the live columns list that are the inputs to the current
  // kernel. Starts from the second evalutor (index 1)
  std::vector<std::vector<i32>> column_mapping =
      analysis_results.column_mapping;


  VLOG(1) << "firebb: op" << ops.at(0).name();
  // Break up kernels into groups that run on the same device
  std::vector<OpArgGroup> groups;
  if (!kernel_factories.empty()) {
    bool first_op = true;
    DeviceType last_device_type;
    i32 kg = 0;
    i32 op_counter = 0; 
    for (size_t i = 1; i < kernel_factories.size() - 1; ++i) {
      KernelFactory* factory = kernel_factories[i];
      /*
       * firebb: hack to ignore input since remapped
       */
      if (ops.at(i).name() == INPUT_OP_NAME)
        continue;
      // Factory is nullptr when we are on a builtin op
      VLOG(1) << "firebb: op" << ops.at(i).name();
      groups.emplace_back();
      auto& op_group = groups.back().op_names;
      auto& op_sampling = groups.back().sampling_args;
      auto& group = groups.back().kernel_factories;
      auto& dc = groups.back().dead_columns;
      auto& uo = groups.back().unused_outputs;
      auto& cm = groups.back().column_mapping;
      auto& st = groups.back().kernel_stencils;
      auto& bt = groups.back().kernel_batch_sizes;
      const std::string& op_name = ops.at(i).name();
      op_group.push_back(op_name);
      if (analysis_results.slice_ops.count(i) > 0) {
        i64 local_op_idx = group.size();
        // Set sampling args
        auto& slice_outputs_per_job =
            groups.back().slice_output_rows[local_op_idx];
        for (auto& job_slice_outputs : analysis_results.slice_output_rows) {
          auto& slice_groups = job_slice_outputs.at(i);
          slice_outputs_per_job.push_back(slice_groups);
        }
      }
      if (analysis_results.unslice_ops.count(i) > 0) {
        i64 local_op_idx = group.size();
        // Set sampling args
        auto& unslice_inputs_per_job =
            groups.back().unslice_input_rows[local_op_idx];
        for (auto& job_unslice_inputs : analysis_results.unslice_input_rows) {
          auto& slice_groups = job_unslice_inputs.at(i);
          unslice_inputs_per_job.push_back(slice_groups);
        }
      }
      if (analysis_results.sampling_ops.count(i) > 0) {
        i64 local_op_idx = group.size();
        // Set sampling args
        auto& sampling_args_per_job = groups.back().sampling_args[local_op_idx];
        for (auto& job : jobs) {
          for (auto& saa : job.sampling_args_assignment()) {
            if (saa.op_index() == i) {
              sampling_args_per_job.emplace_back(
                  saa.sampling_args().begin(),
                  saa.sampling_args().end());
              break;
            }
          }
        }
        assert(sampling_args_per_job.size() == jobs.size());
      }
      group.push_back(std::make_tuple(factory, kernel_configs[i]));
      dc.push_back(dead_columns[kg]);
      uo.push_back(unused_outputs[kg]);
      cm.push_back(column_mapping[kg]);
      st.push_back(analysis_results.stencils[i]);
      bt.push_back(analysis_results.batch_sizes[i]);
      groups.back().task_streams_start = op_counter;
      kg++;
    }
    sArgs.num_ops = op_counter;
  }
  VLOG(1) << "firebb: op" << ops.at(kernel_factories.size() - 1).name();

  i32 num_kernel_groups = static_cast<i32>(groups.size());
  VLOG(1) << "firebb num of kernel groups: " << num_kernel_groups;
  assert(num_kernel_groups > 0);  // is this actually necessary?

  // If ki per node is -1, we set a smart default. Currently, we calculate the
  // maximum possible kernel instances without oversubscribing any part of the
  // pipeline, either CPU or GPU.
  bool has_gpu_kernel = false;
  if (pipeline_instances_per_node == -1) {
    pipeline_instances_per_node = std::numeric_limits<i32>::max();
    for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
      auto& group = groups[kg].kernel_factories;
      for (i32 k = 0; k < group.size(); ++k) {
        // Skip builtin ops
        if (std::get<0>(group[k]) == nullptr) {
          continue;
        }
        KernelFactory* factory = std::get<0>(group[k]);
        DeviceType device_type = factory->get_device_type();
        i32 max_devices = factory->get_max_devices();
        if (max_devices == Kernel::UnlimitedDevices) {
          pipeline_instances_per_node = 1;
        } else {
          pipeline_instances_per_node =
              std::min(pipeline_instances_per_node,
                       device_type == DeviceType::CPU
                           ? db_params_.num_cpus / local_total / max_devices
                           : (i32)num_gpus / max_devices);
        }
        if (device_type == DeviceType::GPU) {
          has_gpu_kernel = true;
        }
      }
    }
    if (pipeline_instances_per_node == std::numeric_limits<i32>::max()) {
      pipeline_instances_per_node = 1;
    }
  }

  if (pipeline_instances_per_node <= 0) {
    RESULT_ERROR(job_result,
                 "BulkJobParameters.pipeline_instances_per_node must -1 for "
                 "auto-default or "
                 " greater than 0 for manual configuration.");
    return grpc::Status::OK;
  }

  // Set up memory pool if different than previous memory pool
  if (!memory_pool_initialized_ ||
      job_params->memory_pool_config() != cached_memory_pool_config_) {
    if (db_params_.num_cpus < local_total * pipeline_instances_per_node &&
        job_params->memory_pool_config().cpu().use_pool()) {
      RESULT_ERROR(job_result,
                   "Cannot oversubscribe CPUs and also use CPU memory pool");
      return grpc::Status::OK;
    }
    if (db_params_.gpu_ids.size() < local_total * pipeline_instances_per_node &&
        job_params->memory_pool_config().gpu().use_pool()) {
      RESULT_ERROR(job_result,
                   "Cannot oversubscribe GPUs and also use GPU memory pool");
      return grpc::Status::OK;
    }
    if (memory_pool_initialized_) {
      destroy_memory_allocators();
    }
    init_memory_allocators(job_params->memory_pool_config(), gpu_ids);
    cached_memory_pool_config_ = job_params->memory_pool_config();
    memory_pool_initialized_ = true;
  }

  omp_set_num_threads(std::thread::hardware_concurrency());

  // Setup shared resources for distributing work to processing threads
  i64 accepted_tasks = 0;
  LoadInputQueue load_work;
  std::vector<EvalQueue> initial_eval_work(pipeline_instances_per_node);
  std::vector<std::vector<EvalQueue>> eval_work(pipeline_instances_per_node);
  OutputEvalQueue output_eval_work(pipeline_instances_per_node);
  std::vector<SaveInputQueue> save_work(db_params_.num_save_workers);
  SaveOutputQueue retired_tasks;

  // Setup load workers
  i32 num_load_workers = db_params_.num_load_workers;
  std::vector<Profiler> load_thread_profilers;
  for (i32 i = 0; i < num_load_workers; ++i) {
    load_thread_profilers.emplace_back(Profiler(base_time));
  }
  std::vector<std::thread> load_threads;
  for (i32 i = 0; i < num_load_workers; ++i) {
    LoadWorkerArgs args{// Uniform arguments
                        node_id_,
                        // Per worker arguments
                        i, db_params_.storage_config, load_thread_profilers[i],
                        job_params->load_sparsity_threshold(), io_packet_size,
                        work_packet_size};

    load_threads.emplace_back(load_driver, std::ref(load_work),
                              std::ref(initial_eval_work), args);
  }

  // Set up all other queues
  std::vector<std::vector<proto::Result>> eval_results(
      pipeline_instances_per_node);

  std::vector<std::tuple<EvalQueue*, EvalQueue*>> pre_eval_queues;
  std::vector<PreEvaluateWorkerArgs> pre_eval_args;
  std::vector<std::vector<std::tuple<EvalQueue*, EvalQueue*>>> eval_queues(
      pipeline_instances_per_node);
  std::vector<std::vector<EvaluateWorkerArgs>> eval_args(
      pipeline_instances_per_node);
  std::vector<std::tuple<EvalQueue*, OutputEvalQueue*>> post_eval_queues;
  std::vector<PostEvaluateWorkerArgs> post_eval_args;

  // For worker threads
  std::vector<std::vector<EvaluateWorker *>> pipeline_eval_workers(
      pipeline_instances_per_node);

  i32 next_cpu_num = 0;
  i32 next_gpu_idx = 0;
  std::mutex startup_lock;
  std::condition_variable startup_cv;
  i32 startup_count = 0;
  i32 eval_total = 0;

  std::vector<Profiler> eval_thread_profilers;
  std::vector<Profiler> pre_eval_profilers;
  std::vector<Profiler> post_eval_profilers;
  // Set up eval_threads profilers
  for (i32 wid = 0; wid < num_eval_threads; wid ++) {
    eval_thread_profilers.push_back(Profiler(base_time));
  }

  // Set up pipeline pre post profiler
  for (i32 pu = 0; pu < pipeline_instances_per_node; pu ++) {
    pre_eval_profilers.push_back(Profiler(base_time));
    post_eval_profilers.push_back(Profiler(base_time));
  }

  for (i32 ki = 0; ki < pipeline_instances_per_node; ++ki) {
    auto& work_queues = eval_work[ki];

    std::vector<proto::Result>& results = eval_results[ki];
    work_queues.resize(num_kernel_groups); 
    results.resize(num_kernel_groups);
    for (auto& result : results) {
      result.set_success(true);
    }

    // Evaluate worker
    DeviceHandle first_kernel_type;
    for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
      // All stages are free in the beginning
      pipeline_status[ki].push_back(false);

      auto& group = groups[kg].kernel_factories;
      std::vector<EvaluateWorkerArgs>& thread_args = eval_args[ki];
      // HACK(apoms): we assume all ops in a kernel group use the
      //   same number of devices for now.
      KernelFactory* factory = nullptr;
      for (size_t i = 0; i < group.size(); ++i) {
        if (std::get<0>(group[i]) != nullptr) {
          factory = std::get<0>(group[i]) ;
        }
      }
      DeviceType device_type = DeviceType::CPU;
      i32 max_devices = 1;
      // Factory should only be null if we only have builtin ops
      if (factory != nullptr) {
        device_type = factory->get_device_type();
        max_devices = factory->get_max_devices();
      }
      if (device_type == DeviceType::CPU) {
        for (i32 i = 0; i < max_devices; ++i) {
          i32 device_id = 0;
          next_cpu_num++ % num_cpus;
          for (size_t i = 0; i < group.size(); ++i) {
            KernelConfig& config = std::get<1>(group[i]);
            config.devices.clear();
            config.devices.push_back({device_type, device_id});
          }
        }
      } else {
        for (i32 i = 0; i < max_devices; ++i) {
          i32 device_id = gpu_ids[next_gpu_idx++ % num_gpus];
          for (size_t i = 0; i < group.size(); ++i) {
            KernelConfig& config = std::get<1>(group[i]);
            config.devices.clear();
            config.devices.push_back({device_type, device_id});
          }
        }
      }
      // Get the device handle for the first kernel in the pipeline
      if (kg == 0) {
        first_kernel_type = std::get<1>(group[0]).devices[0];
      }

      thread_args.emplace_back(EvaluateWorkerArgs{
          // Uniform arguments
          node_id_, startup_lock, startup_cv, startup_count,

          // Per worker arguments
          // hack(firebb) hack profiler. TODO: use per eval thread
          // profiler instead of kernel group profiler
          ki, kg, groups[kg], scheduler_profiler, results[kg]});
      eval_total += 1;
    }
    // Pre evaluate worker
    {
      EvalQueue* input_work_queue;
      if (distribute_work_dynamically) {
        input_work_queue = &initial_eval_work[ki];
      } else {
        input_work_queue = &initial_eval_work[0];
      }
      //EvalQueue* output_work_queue =
      //    &work_queues[0];
      EvalQueue* output_work_queue =
          &pre_output_queues[ki];
      assert(groups.size() > 0);
      pre_eval_queues.push_back(
          std::make_tuple(input_work_queue, output_work_queue));
      DeviceHandle decoder_type = std::getenv("FORCE_CPU_DECODE")
        ? CPU_DEVICE
        : first_kernel_type;
      pre_eval_args.emplace_back(PreEvaluateWorkerArgs{
          // Uniform arguments
          node_id_, num_cpus, job_params->work_packet_size(),

          // Per worker arguments
          ki, decoder_type, std::ref(pre_eval_profilers[ki]),
      });
    }

    // Post evaluate worker
    {
      auto& output_op = ops.at(ops.size() - 1);
      std::vector<std::string> column_names;
      for (auto& op_input : output_op.inputs()) {
        column_names.push_back(op_input.column());
      }

      EvalQueue* input_work_queue = &post_input_queues[ki];
      OutputEvalQueue* output_work_queue = &output_eval_work;
      post_eval_queues.push_back(
          std::make_tuple(input_work_queue, output_work_queue));
      post_eval_args.emplace_back(PostEvaluateWorkerArgs{
          // Uniform arguments
          node_id_,

          // Per worker arguments
          ki, std::ref(post_eval_profilers[ki]), column_mapping.back(),
          final_output_columns, final_compression_options,
      });
    }
  }

  // Launch eval worker threads
  std::vector<std::thread> pre_eval_threads;
  //std::vector<std::vector<std::thread>> eval_threads;
  std::vector<std::thread> eval_threads;
  std::vector<std::thread> post_eval_threads;
  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
    // Pre thread
    pre_eval_threads.emplace_back(
        pre_evaluate_driver, std::ref(*std::get<0>(pre_eval_queues[pu])),
        std::ref(*std::get<1>(pre_eval_queues[pu])), pre_eval_args[pu]);
    // Op threads
    //eval_threads.emplace_back();
    //std::vector<std::thread>& threads = eval_threads.back();
    std::vector<EvaluateWorker *>& workers = pipeline_eval_workers[pu];
    for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
      //threads.emplace_back(
      //    evaluate_driver, std::ref(*std::get<0>(eval_queues[pu][kg])),
      //    std::ref(*std::get<1>(eval_queues[pu][kg])), eval_args[pu][kg]);
      EvaluateWorkerArgs& args = std::ref(eval_args[pu][kg]);
      workers.push_back(new EvaluateWorker(args));
    }
    // Post threads
    post_eval_threads.emplace_back(
        post_evaluate_driver, std::ref(*std::get<0>(post_eval_queues[pu])),
        std::ref(*std::get<1>(post_eval_queues[pu])), post_eval_args[pu]);
  }

  std::thread scheduler_thread(schedule, sArgs);

  for (i32 wid = 0; wid < num_eval_threads; ++wid) {
    eval_threads.emplace_back(worker_thread,
        std::ref(task_queues[wid]),
        std::ref(result_queue),
        std::ref(pipeline_eval_workers),
        std::ref(eval_args),
        std::ref(eval_thread_profilers[wid]),
        wid);
  }

  // Setup save coordinator
  std::thread save_coordinator_thread(
      save_coordinator, std::ref(output_eval_work), std::ref(save_work));

  // Setup save workers
  i32 num_save_workers = db_params_.num_save_workers;
  std::vector<Profiler> save_thread_profilers;
  for (i32 i = 0; i < num_save_workers; ++i) {
    save_thread_profilers.emplace_back(Profiler(base_time));
  }
  std::vector<std::thread> save_threads;
  for (i32 i = 0; i < num_save_workers; ++i) {
    SaveWorkerArgs args{// Uniform arguments
                        node_id_,
                        // Per worker arguments
                        i, db_params_.storage_config, save_thread_profilers[i]};

    save_threads.emplace_back(save_driver, std::ref(save_work[i]),
                              std::ref(retired_tasks), args);
  }

  if (job_params->profiling()) {
    // Wait until all evaluate workers have started up
    std::unique_lock<std::mutex> lk(startup_lock);
    startup_cv.wait(lk, [&] {
      return eval_total == startup_count;
    });
  }
  VLOG(1) << "worker all started!";

  timepoint_t start_time = now();

  // Monitor amount of work left and request more when running low
  // Round robin work
  std::vector<i64> allocated_work_to_queues(pipeline_instances_per_node);
  std::vector<i64> retired_work_for_queues(pipeline_instances_per_node);
  bool finished = false;
  while (true) {
    if (trigger_shutdown_.raised()) {
      // Abandon ship!
      VLOG(1) << "Worker " << node_id_ << " received shutdown while in NewJob";
      RESULT_ERROR(job_result, "Worker %d shutdown while processing NewJob",
                   node_id_);
      break;
    }
    // We batch up retired tasks to avoid sync overhead
    std::vector<std::tuple<i32, i64, i64>> batched_retired_tasks;
    while (retired_tasks.size() > 0) {
      // Pull retired tasks
      std::tuple<i32, i64, i64> task_retired;
      retired_tasks.pop(task_retired);
      batched_retired_tasks.push_back(task_retired);
    }
    if (!batched_retired_tasks.empty()) {
      // Make sure the retired tasks were flushed to disk before confirming
      std::fflush(NULL);
      sync();
    }
    for (std::tuple<i32, i64, i64>& task_retired : batched_retired_tasks) {
      // Inform master that this task was finished
      grpc::ClientContext context;
      proto::FinishedWorkParameters params;
      proto::Empty empty;

      params.set_node_id(node_id_);
      params.set_job_id(std::get<1>(task_retired));
      params.set_task_id(std::get<2>(task_retired));
      grpc::Status status = master_->FinishedWork(&context, params, &empty);

      if (!status.ok()) {
        RESULT_ERROR(job_result,
                     "Worker %d could not tell finished work to master",
                     node_id_);
        break;
      }

      // Update how much is in each pipeline instances work queue
      retired_work_for_queues[std::get<0>(task_retired)] += 1;
    }
    i64 total_tasks_processed = 0;
    for (i64 t : retired_work_for_queues) {
      total_tasks_processed += t;
    }
    if (finished) {
      if (total_tasks_processed == accepted_tasks) {
        break;
      } else {
        std::this_thread::yield();
        continue;
      }
    }
    i32 local_work = accepted_tasks - total_tasks_processed;
    if (local_work <
        pipeline_instances_per_node * job_params->tasks_in_queue_per_pu()) {
      grpc::ClientContext context;
      proto::NodeInfo node_info;
      proto::NewWork new_work;

      node_info.set_node_id(node_id_);
      grpc::Status status = master_->NextWork(&context, node_info, &new_work);
      if (!status.ok()) {
        RESULT_ERROR(job_result,
                     "Worker %d could not get next work from master", node_id_);
        break;
      }

      if (new_work.no_more_work()) {
        // No more work left
        VLOG(1) << "Node " << node_id_ << " received done signal.";
        finished = true;
      } else {
        // Perform analysis on load work entry to determine upstream
        // requirements and when to discard elements.
        std::deque<TaskStream> task_stream;
        LoadWorkEntry stenciled_entry;
        derive_stencil_requirements(
            meta, table_meta, jobs.at(new_work.job_index()), ops,
            analysis_results, job_params->boundary_condition(),
            new_work.table_id(), new_work.job_index(), new_work.task_index(),
            std::vector<i64>(new_work.output_rows().begin(),
                             new_work.output_rows().end()),
            stenciled_entry, task_stream);

        // Determine which worker to allocate to
        i32 target_work_queue = -1;
        i32 min_work = std::numeric_limits<i32>::max();
        for (int i = 0; i < pipeline_instances_per_node; ++i) {
          i64 outstanding_work =
              allocated_work_to_queues[i] - retired_work_for_queues[i];
          if (outstanding_work < min_work) {
            min_work = outstanding_work;
            target_work_queue = i;
          }
        }
        load_work.push(
            std::make_tuple(target_work_queue, task_stream, stenciled_entry));
        allocated_work_to_queues[target_work_queue]++;
        accepted_tasks++;
      }
    }

    for (size_t i = 0; i < eval_results.size(); ++i) {
      for (size_t j = 0; j < eval_results[i].size(); ++j) {
        auto& result = eval_results[i][j];
        if (!result.success()) {
          LOG(WARNING) << "(N/KI/KG: " << node_id_ << "/" << i << "/" << j
                       << ") returned error result: " << result.msg();
          job_result->set_success(false);
          job_result->set_msg(result.msg());
          goto leave_loop;
        }
      }
    }
    goto remain_loop;
  leave_loop:
    break;
  remain_loop:

    std::this_thread::yield();
  }

  // If the job failed, can't expect queues to have drained, so
  // attempt to flush all queues here (otherwise we could block
  // on pushing into a queue)
  if (!job_result->success()) {
    load_work.clear();
    for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
      initial_eval_work[pu].clear();
    }
    for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
      for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
        eval_work[pu][kg].clear();
      }
    }
    for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
      post_input_queues[pu].clear();
    }
    output_eval_work.clear();
    for (i32 i = 0; i < num_save_workers; ++i) {
      save_work[i].clear();
    }
    // TODO: clear pre_out and post_in
  }

  auto push_worker_thread_exit_message = [](IntermediateQueue& q) {
    Intermediate inter;
    inter.is_last = true;
    q.push(inter);
  };

  auto push_exit_message = [](EvalQueue& q) {
    EvalWorkEntry entry;
    entry.job_index = -1;
    q.push(std::make_tuple(std::deque<TaskStream>(), entry));
  };

  auto push_output_eval_exit_message = [](OutputEvalQueue& q) {
    EvalWorkEntry entry;
    entry.job_index = -1;
    q.push(std::make_tuple(0, entry));
  };

  auto push_save_exit_message = [](SaveInputQueue& q) {
    EvalWorkEntry entry;
    entry.job_index = -1;
    q.push(std::make_tuple(0, entry));
  };

  // Push sentinel work entries into queue to terminate load threads
  for (i32 i = 0; i < num_load_workers; ++i) {
    LoadWorkEntry entry;
    entry.set_job_index(-1);
    load_work.push(
        std::make_tuple(0, std::deque<TaskStream>(), entry));
  }

  for (i32 i = 0; i < num_load_workers; ++i) {
    // Wait until all load threads have finished
    load_threads[i].join();
  }

  // Push sentinel work entries into queue to terminate eval threads
  for (i32 i = 0; i < pipeline_instances_per_node; ++i) {
    if (distribute_work_dynamically) {
      push_exit_message(initial_eval_work[i]);
    } else {
      push_exit_message(initial_eval_work[0]);
    }
  }

  for (i32 i = 0; i < pipeline_instances_per_node; ++i) {
    // Wait until pre eval has finished
    pre_eval_threads[i].join();
  }

  //for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
  //  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
  //    push_exit_message(eval_work[pu][kg]);
  //  }
  //  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
  //    // Wait until eval has finished
  //    eval_threads[pu][kg].join();
  //  }
  //}
  
  VLOG(1) << "signal post_eval finish!";
  // Terminate post eval threads
  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
    push_exit_message(post_input_queues[pu]);
  }
  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
    // Wait until eval has finished
    post_eval_threads[pu].join();
  }
  VLOG(1) << "post eval finished";

  // Push sentinel work entries into queue to terminate coordinator thread
  push_output_eval_exit_message(output_eval_work);
  save_coordinator_thread.join();

  // Push sentinel work entries into queue to terminate save threads
  for (i32 i = 0; i < num_save_workers; ++i) {
    push_save_exit_message(save_work[i]);
  }
  for (i32 i = 0; i < num_save_workers; ++i) {
    // Wait until eval has finished
    save_threads[i].join();
  }

  // Terminate worker threads
  VLOG(1) << "signal worker finish!";
  for (i32 i = 0; i < num_eval_threads; i++) {
    push_worker_thread_exit_message(task_queues[i]);
  }
  for (i32 i = 0; i < num_eval_threads; i++) {
    eval_threads[i].join();
  }

  // Terminate scheduler
  scheduler_thread.join();

  // Ensure all files are flushed
  if (job_params->profiling()) {
    std::fflush(NULL);
    sync();
  }

  if (!job_result->success()) {
    return grpc::Status::OK;
  }

  // Write out total time interval
  timepoint_t end_time = now();

  // Execution done, write out profiler intervals for each worker
  // TODO: job_name -> job_id?
  i32 job_id = meta.get_bulk_job_id(job_params->job_name());
  std::string profiler_file_name = bulk_job_profiler_path(job_id, node_id_);
  std::unique_ptr<WriteFile> profiler_output;
  BACKOFF_FAIL(
      make_unique_write_file(storage_, profiler_file_name, profiler_output));

  i64 base_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(base_time)
          .time_since_epoch()
          .count();
  i64 start_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(start_time)
          .time_since_epoch()
          .count();
  i64 end_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(end_time)
          .time_since_epoch()
          .count();
  s_write(profiler_output.get(), start_time_ns);
  s_write(profiler_output.get(), end_time_ns);

  i64 out_rank = node_id_;
  // Load worker profilers
  u8 load_worker_count = num_load_workers + num_eval_threads;
  s_write(profiler_output.get(), load_worker_count);
  for (i32 i = 0; i < num_load_workers; ++i) {
    write_profiler_to_file(profiler_output.get(), out_rank, "load", "", i,
                           load_thread_profilers[i]);
  }

  // Evaluate worker profilers
  //u8 eval_worker_count = num_eval_threads;
  //s_write(profiler_output.get(), eval_worker_count);
  for (i32 wid = 0; wid < num_eval_threads; wid ++) {
    std::string tag = "worker";
    write_profiler_to_file(profiler_output.get(), out_rank, "eval", tag, wid,
                             eval_thread_profilers[wid]);
  }

  // Pre/Post Evaluate worker profilers
  u8 prepost_worker_count = pipeline_instances_per_node;
  s_write(profiler_output.get(), prepost_worker_count);
  u8 profilers_per_chain = 2;
  s_write(profiler_output.get(), profilers_per_chain);
  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
    i32 i = pu;
    {
      std::string tag = "pre";
      write_profiler_to_file(profiler_output.get(), out_rank, "eval", tag, i,
                             pre_eval_profilers[pu]);
    }
    {
      std::string tag = "post";
      write_profiler_to_file(profiler_output.get(), out_rank, "eval", tag, i,
                             post_eval_profilers[pu]);
    }
  }

  // Save worker profilers
  u8 save_worker_count = num_save_workers;
  s_write(profiler_output.get(), save_worker_count);
  for (i32 i = 0; i < num_save_workers; ++i) {
    write_profiler_to_file(profiler_output.get(), out_rank, "save", "", i,
                           save_thread_profilers[i]);
  }

  BACKOFF_FAIL(profiler_output->save());

  std::fflush(NULL);
  sync();

  VLOG(1) << "Worker " << node_id_ << " finished NewJob";

  // Set to idle if we finished without a shutdown
  state_.test_and_set(RUNNING_JOB, IDLE);

  return grpc::Status::OK;
}

grpc::Status WorkerImpl::LoadOp(grpc::ServerContext* context,
                                const proto::OpPath* op_path,
                                proto::Empty* empty) {
  const std::string& so_path = op_path->path();
  VLOG(1) << "Worker " << node_id_ << " loading Op library: " << so_path;
  void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  LOG_IF(FATAL, handle == nullptr)
      << "dlopen of " << so_path << " failed: " << dlerror();
  return grpc::Status::OK;
}

grpc::Status WorkerImpl::RegisterOp(
    grpc::ServerContext* context, const proto::OpRegistration* op_registration,
    proto::Result* result) {
  const std::string& name = op_registration->name();
  const bool variadic_inputs = op_registration->variadic_inputs();
  std::vector<Column> input_columns;
  size_t i = 0;
  for (auto& c : op_registration->input_columns()) {
    Column col;
    col.set_id(i++);
    col.set_name(c.name());
    col.set_type(c.type());
    input_columns.push_back(col);
  }
  std::vector<Column> output_columns;
  i = 0;
  for (auto& c : op_registration->output_columns()) {
    Column col;
    col.set_id(i++);
    col.set_name(c.name());
    col.set_type(c.type());
    output_columns.push_back(col);
  }
  bool can_stencil = op_registration->can_stencil();
  std::vector<i32> stencil(op_registration->preferred_stencil().begin(),
                                 op_registration->preferred_stencil().end());
  if (stencil.empty()) {
    stencil = {0};
  }
  bool has_bounded_state = op_registration->has_bounded_state();
  i32 warmup = op_registration->warmup();
  bool has_unbounded_state = op_registration->has_unbounded_state();
  OpInfo* info = new OpInfo(name, variadic_inputs, input_columns,
                            output_columns, can_stencil, stencil,
                            has_bounded_state, warmup, has_unbounded_state);
  OpRegistry* registry = get_op_registry();
  registry->add_op(name, info);
  VLOG(1) << "Worker " << node_id_ << " registering Op: " << name;

  return grpc::Status::OK;
}

grpc::Status WorkerImpl::RegisterPythonKernel(
    grpc::ServerContext* context,
    const proto::PythonKernelRegistration* python_kernel,
    proto::Result* result) {
  const std::string& op_name = python_kernel->op_name();
  DeviceType device_type = python_kernel->device_type();
  const std::string& kernel_str = python_kernel->kernel_str();
  const std::string& pickled_config = python_kernel->pickled_config();
  // Create a kernel builder function
  auto constructor = [kernel_str, pickled_config](const KernelConfig& config) {
    return new PythonKernel(config, kernel_str, pickled_config);
  };
  // Create a new kernel factory
  KernelFactory* factory =
      new KernelFactory(op_name, device_type, 1, false, 1, constructor);
  // Register the kernel
  KernelRegistry* registry = get_kernel_registry();
  registry->add_kernel(op_name, factory);
  VLOG(1) << "Worker " << node_id_ << " registering Python Kernel: " << op_name;
  return grpc::Status::OK;
}

grpc::Status WorkerImpl::Shutdown(grpc::ServerContext* context,
                                  const proto::Empty* empty, Result* result) {
  State state = state_.get();
  switch (state) {
    case RUNNING_JOB: {
      // trigger_shutdown will inform job to stop working
      break;
    }
    case SHUTTING_DOWN: {
      // Already shutting down
      result->set_success(true);
      return grpc::Status::OK;
    }
    case INITIALIZING: {
      break;
    }
    case IDLE: {
      break;
    }
  }
  state_.set(SHUTTING_DOWN);
  try_unregister();
  // Inform watchdog that we are done for
  trigger_shutdown_.set();
  result->set_success(true);
  return grpc::Status::OK;
}

grpc::Status WorkerImpl::PokeWatchdog(grpc::ServerContext* context,
                                      const proto::Empty* empty,
                                      proto::Empty* result) {
  watchdog_awake_ = true;
  return grpc::Status::OK;
}

grpc::Status WorkerImpl::Ping(grpc::ServerContext* context,
                              const proto::Empty* empty1,
                              proto::Empty* empty2) {
  return grpc::Status::OK;
}

void WorkerImpl::start_watchdog(grpc::Server* server, bool enable_timeout,
                                i32 timeout_ms) {
  watchdog_thread_ = std::thread([this, server, enable_timeout, timeout_ms]() {
    double time_since_check = 0;
    // Wait until shutdown is triggered or watchdog isn't woken up
    if (!enable_timeout) {
      trigger_shutdown_.wait();
    }
    while (!trigger_shutdown_.raised()) {
      auto sleep_start = now();
      trigger_shutdown_.wait_for(timeout_ms);
      time_since_check += nano_since(sleep_start) / 1e6;
      if (time_since_check > timeout_ms) {
        if (!watchdog_awake_) {
          // Watchdog not woken, time to bail out
          LOG(ERROR) << "Worker did not receive heartbeat in " << timeout_ms
                     << "ms. Shutting down.";
          trigger_shutdown_.set();
        }
        watchdog_awake_ = false;
        time_since_check = 0;
      }
    }
    // Shutdown self
    server->Shutdown();
  });
}

void WorkerImpl::register_with_master() {
  assert(state_.get() == State::INITIALIZING);

  VLOG(1) << "Worker try to register with master";

  grpc::ClientContext context;
  proto::WorkerParams worker_info;
  proto::Registration registration;

  worker_info.set_port(worker_port_);
  proto::MachineParameters* params = worker_info.mutable_params();
  params->set_num_cpus(db_params_.num_cpus);
  params->set_num_load_workers(db_params_.num_cpus);
  params->set_num_save_workers(db_params_.num_cpus);
  for (i32 gpu_id : db_params_.gpu_ids) {
    params->add_gpu_ids(gpu_id);
  }

  grpc::Status status =
      master_->RegisterWorker(&context, worker_info, &registration);
  LOG_IF(FATAL, !status.ok())
      << "Worker could not contact master server at " << master_address_ << " ("
      << status.error_code() << "): " << status.error_message();
  VLOG(1) << "Worker registered with master";

  node_id_ = registration.node_id();

  state_.set(State::IDLE);
}

void WorkerImpl::try_unregister() {
  if (state_.get() != State::INITIALIZING && !unregistered_.test_and_set()) {
    grpc::ClientContext ct;
    proto::NodeInfo node_info;
    proto::Empty em;
    node_info.set_node_id(node_id_);
    grpc::Status status = master_->UnregisterWorker(&ct, node_info, &em);
    if (!status.ok()) {
      VLOG(1) << "Worker could not unregister from master server "
              << "(" << status.error_code() << "): " << status.error_message();
    }
  }
}

}
}
