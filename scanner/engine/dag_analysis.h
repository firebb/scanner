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

#include "scanner/engine/metadata.h"
#include "scanner/engine/table_meta_cache.h"
#include "scanner/engine/runtime.h"

#include <deque>

namespace scanner {
namespace internal {

const std::string INPUT_OP_NAME = "Input";
const std::string OUTPUT_OP_NAME = "OutputTable";
const std::string SAMPLE_OP_NAME = "Sample";
const std::string SPACE_OP_NAME = "Space";
const std::string SLICE_OP_NAME = "Slice";
const std::string UNSLICE_OP_NAME = "Unslice";

const std::vector<std::string> BUILTIN_OP_NAMES = {
  INPUT_OP_NAME,
  OUTPUT_OP_NAME,
  SAMPLE_OP_NAME,
  SPACE_OP_NAME,
  SLICE_OP_NAME,
  UNSLICE_OP_NAME,
};

bool is_builtin_op(const std::string& name);

class OpStage {
  public:
    OpStage(i32 kg)
      : kg_(kg) {}
    i32 kg() {return kg_;}
    i32 num_inputs() {return num_inputs_;}
    void set_inputs(i32 num) {num_inputs_ = num;}
    void add_op(i32 op_idx) {ops.push_back(op_idx);}
    void add_child(i32 kg) {children.push_back(kg);}
    bool is_last() { 
      return find(children.begin(), children.end(), -1) == children.end();
    }
    
    // OP indexes in the stage. TODO: one stage may contains multiple ops.
    std::vector<i32> ops;
    std::vector<i32> children;
    // Output column index -> [<next stage, input column index>]
    std::vector<std::vector<std::pair<i32, i32>>> output_mapping;
  private:
    i32 kg_;
    i32 num_inputs_;
};

struct DAGAnalysisInfo {
  std::vector<i32> op_slice_level;
  std::map<i64, i64> input_ops;
  std::map<i64, i64> slice_ops;
  std::map<i64, i64> unslice_ops;
  std::map<i64, i64> sampling_ops;
  std::map<i64, std::vector<i64>> op_children;

  // Input rows to slice Ops per Job
  std::vector<std::map<i64, i64>> slice_input_rows;
  // Job -> Op -> Slice
  std::vector<std::map<i64, std::vector<i64>>> slice_output_rows;
  // Input rows to unslice Ops per Job
  // Job -> Op -> Slice
  std::vector<std::map<i64, std::vector<i64>>> unslice_input_rows;
  // Total rows for each ops domain
  // Job -> Op -> Slice
  std::vector<std::map<i64, std::vector<i64>>> total_rows_per_op;
  // Total output rows per Job
  std::vector<i64> total_output_rows;

  std::map<i64, bool> bounded_state_ops;
  std::map<i64, bool> unbounded_state_ops;
  std::map<i64, i32> warmup_sizes;
  std::map<i64, i32> batch_sizes;
  std::map<i64, std::vector<i32>> stencils;

  // Filled in by remap_input_op_edges
  std::map<i64, i64> input_ops_to_first_op_columns;

  // Op -> Columns
  std::vector<std::vector<std::tuple<i32, std::string>>> live_columns;
  std::vector<std::vector<i32>> dead_columns;
  std::vector<std::vector<i32>> unused_outputs;
  std::vector<std::vector<i32>> column_mapping;
};


Result validate_jobs_and_ops(
    DatabaseMetadata& meta, TableMetaCache& table_metas,
    const std::vector<proto::Job>& jobs,
    const std::vector<proto::Op>& ops,
    DAGAnalysisInfo& info);

Result determine_input_rows_to_slices(
    DatabaseMetadata& meta, TableMetaCache& table_metas,
    const std::vector<proto::Job>& jobs,
    const std::vector<proto::Op>& ops,
    DAGAnalysisInfo& info);

Result derive_slice_final_output_rows(
    const proto::Job& job,
    const std::vector<proto::Op>& ops,
    i64 slice_op_idx,
    i64 slice_input_rows,
    DAGAnalysisInfo& info,
    std::vector<i64>& slice_output_partition);

void populate_analysis_info(const std::vector<proto::Op>& ops,
                            DAGAnalysisInfo& info);

// Change all edges from input Ops to instead come from the first Op.
// We currently only implement IO at the start and end of a pipeline.
void remap_input_op_edges(std::vector<proto::Op>& ops,
                          DAGAnalysisInfo& info);

void perform_liveness_analysis(const std::vector<proto::Op>& ops,
                               DAGAnalysisInfo& info);

void perform_liveness_analysis(const std::vector<proto::Op>& ops,
    DAGAnalysisInfo& info,
    std::vector<OpStage> &pipeline_stages,
    std::map<i32, i32> &op_stage_mapping,
    std::vector<std::vector<std::pair<i32, i32>>> &input_col_mapping);

Result derive_stencil_requirements(
    const DatabaseMetadata& meta, const TableMetaCache& table_meta,
    const proto::Job& job, const std::vector<proto::Op>& ops,
    const DAGAnalysisInfo& analysis_results,
    proto::BulkJobParameters::BoundaryCondition boundary_condition,
    i64 table_id, i64 job_idx, i64 task_idx,
    const std::vector<i64>& output_rows, LoadWorkEntry& output_entry,
    std::deque<TaskStream>& task_streams);

// Result derive_input_rows_from_output_rows(
//     const std::vector<proto::Job>& jobs,
//     const std::vector<proto::Op>& ops,
//     const std::vector<std::vector<i64>>& output_rows,
//     DAGAnalysisInfo& info,
//     std::vector<std::vector<i64>>& input_rows);
}
}
