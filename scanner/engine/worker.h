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

#pragma once

#include "scanner/engine/dag_analysis.h"
#include "scanner/engine/metadata.h"
#include "scanner/engine/rpc.grpc.pb.h"
#include "scanner/engine/runtime.h"

#include <grpc/grpc_posix.h>
#include <grpc/support/log.h>
#include <atomic>
#include <thread>
#include <boost/python.hpp>

namespace scanner {
namespace internal {
struct OpStatus {
  bool running = false;
  bool reserve_by_task = false;
  i32 job_index = -1;
  i32 task_index = -1;
  DeviceType device_type;
  std::vector<i32> device_ids;
};


struct SchedulerArgs{
  // Num gpu devices
  i32 num_gpu_threads;
  // Num worker threads
  i32 num_eval_threads;
  // Num instances
  i32 pipeline_instances_per_node;
  // Num of columns to post eval 
  i32 num_post_col;
  // Num of ops in evaluate stages
  i32 num_ops; 
  // PreEvaluator Queues
  std::vector<EvalQueue> &pre_output_queues;
  // Input col -> [<next stage, input column index>]
  std::vector<std::vector<std::pair<i32, i32>>> &input_col_mapping;
  // PostEvaluator Queues
  std::vector<EvalQueue> &post_input_queues;
  // Intermediate Queue
  IntermediateQueue &result_queue;
  // Assign task Queues
  std::vector<IntermediateQueue> &task_queues;
  // Pipeline stages
  std::vector<OpStage> &pipeline_stages;
  // Pipeline status 
  std::vector<std::vector<OpStatus>> &pipeline_status;
  // Profiler
  Profiler &profiler;

  SchedulerArgs(std::vector<EvalQueue> &pre,
      std::vector<EvalQueue> &post,
      std::vector<std::vector<std::pair<i32, i32>>> &mapping,
      IntermediateQueue &result, 
      std::vector<IntermediateQueue> &task,
      std::vector<OpStage> &stages,
      std::vector<std::vector<OpStatus>> &status,
      Profiler &prof)
   : pre_output_queues(pre),
     post_input_queues(post),
     input_col_mapping(mapping),
     result_queue(result),
     task_queues(task),
     pipeline_stages(stages), 
     pipeline_status(status),
     profiler(prof) {} 
};

struct WorkerThreadArgs {
  i32 id;
};

class WorkerImpl final : public proto::Worker::Service {
 public:
  WorkerImpl(DatabaseParameters& db_params, std::string master_address,
             std::string worker_port);

  ~WorkerImpl();

  grpc::Status NewJob(grpc::ServerContext* context,
                      const proto::BulkJobParameters* job_params,
                      proto::Result* job_result);

  grpc::Status LoadOp(grpc::ServerContext* context,
                      const proto::OpPath* op_path, proto::Empty* empty);

  grpc::Status RegisterOp(grpc::ServerContext* context,
                          const proto::OpRegistration* op_registration,
                          proto::Result* result);

  grpc::Status RegisterPythonKernel(
      grpc::ServerContext* context,
      const proto::PythonKernelRegistration* python_kernel,
      proto::Result* result);

  grpc::Status Shutdown(grpc::ServerContext* context, const proto::Empty* empty,
                        Result* result);

  grpc::Status PokeWatchdog(grpc::ServerContext* context,
                            const proto::Empty* empty, proto::Empty* result);

  grpc::Status Ping(grpc::ServerContext* context, const proto::Empty* empty,
                    proto::Empty* result);

  void start_watchdog(grpc::Server* server, bool enable_timeout,
                      i32 timeout_ms = 50000);

  void register_with_master();

 private:
  void try_unregister();

  enum State {
    INITIALIZING,
    IDLE,
    RUNNING_JOB,
    SHUTTING_DOWN,
  };

  Condition<State> state_;
  std::atomic_flag unregistered_;

  std::thread watchdog_thread_;
  std::atomic<bool> watchdog_awake_;
  std::unique_ptr<proto::Master::Stub> master_;
  storehouse::StorageConfig* storage_config_;
  DatabaseParameters db_params_;
  Flag trigger_shutdown_;
  std::string master_address_;
  std::string worker_port_;
  i32 node_id_;
  storehouse::StorageBackend* storage_;
  std::map<std::string, TableMetadata*> table_metas_;
  bool memory_pool_initialized_ = false;
  MemoryPoolConfig cached_memory_pool_config_;
};
}
}
