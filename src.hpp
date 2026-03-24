#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // For round i (0-based), we need to use keys[0..i] and values[0..i]
    // Q = current_query has shape [(i+1), d] where d=512
    // We need to compute: MatMul(Softmax(MatMul(Q, K^T)), V)

    // Step 1: Concatenate all K[0..i] to form K_all
    Matrix* K_all = matrix_memory_allocator.Allocate("K_all_" + std::to_string(i));
    // Start with first key
    gpu_sim.Copy(keys[0], K_all, Position::kInGpuHbm);

    // Concatenate remaining keys
    for (size_t j = 1; j <= i; ++j) {
      Matrix* temp = matrix_memory_allocator.Allocate("temp_K_concat_" + std::to_string(j));
      gpu_sim.Concat(K_all, keys[j], temp, 0, Position::kInGpuHbm); // axis=0 for vertical concatenation
      gpu_sim.ReleaseMatrix(K_all);
      K_all = temp;
    }

    // Step 2: Concatenate all V[0..i] to form V_all
    Matrix* V_all = matrix_memory_allocator.Allocate("V_all_" + std::to_string(i));
    // Start with first value
    gpu_sim.Copy(values[0], V_all, Position::kInGpuHbm);

    // Concatenate remaining values
    for (size_t j = 1; j <= i; ++j) {
      Matrix* temp = matrix_memory_allocator.Allocate("temp_V_concat_" + std::to_string(j));
      gpu_sim.Concat(V_all, values[j], temp, 0, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(V_all);
      V_all = temp;
    }

    // Step 3: Move matrices to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K_all);
    gpu_sim.MoveMatrixToSharedMem(V_all);

    // Step 4: Transpose K_all in SRAM
    Matrix* K_transpose = matrix_memory_allocator.Allocate("K_transpose_" + std::to_string(i));
    gpu_sim.Transpose(K_all, Position::kInSharedMemory);
    gpu_sim.Copy(K_all, K_transpose, Position::kInSharedMemory);

    // Step 5: Compute Q * K^T
    Matrix* QKt = matrix_memory_allocator.Allocate("QKt_" + std::to_string(i));
    gpu_sim.MatMul(current_query, K_transpose, QKt);

    // Step 6: Compute exp(Q * K^T)
    Matrix* exp_QKt = matrix_memory_allocator.Allocate("exp_QKt_" + std::to_string(i));
    gpu_sim.MatExp(QKt, exp_QKt);

    // Step 7: Compute softmax row by row
    // exp_QKt has shape (i+1) x (i+1)
    size_t n = i + 1; // Matrix dimension for this round

    // Create result matrix for softmax - start with first row
    Matrix* softmax_result = nullptr;

    for (size_t row_idx = 0; row_idx < n; ++row_idx) {
      // Get current row
      Matrix* current_row = matrix_memory_allocator.Allocate("row_" + std::to_string(row_idx));
      gpu_sim.GetRow(exp_QKt, row_idx, current_row, Position::kInSharedMemory);

      // Compute sum of this row
      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(row_idx));
      gpu_sim.Sum(current_row, row_sum); // row_sum is 1x1

      // Divide row by its sum
      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(row_idx));
      gpu_sim.MatDiv(current_row, row_sum, softmax_row);

      // Add this row to result matrix
      if (row_idx == 0) {
        // First row, use as initial result
        softmax_result = softmax_row;
        // Don't release softmax_row, it's now softmax_result
      } else {
        // Concatenate vertically
        Matrix* temp = matrix_memory_allocator.Allocate("temp_concat_" + std::to_string(row_idx));
        gpu_sim.Concat(softmax_result, softmax_row, temp, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_result);
        gpu_sim.ReleaseMatrix(softmax_row);
        softmax_result = temp;
      }

      // Clean up other temporary matrices for this row
      gpu_sim.ReleaseMatrix(current_row);
      gpu_sim.ReleaseMatrix(row_sum);
    }

    // Step 9: Compute softmax_result * V_all
    Matrix* final_result = matrix_memory_allocator.Allocate("final_" + std::to_string(i));
    gpu_sim.MatMul(softmax_result, V_all, final_result);

    // Step 10: Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(final_result);

    // Run the simulator
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Commit answer
    rater.CommitAnswer(*final_result);

    // Clean up (optional, as MatrixMemoryAllocator should handle it)
    gpu_sim.ReleaseMatrix(K_all);
    gpu_sim.ReleaseMatrix(V_all);
    gpu_sim.ReleaseMatrix(K_transpose);
    gpu_sim.ReleaseMatrix(QKt);
    gpu_sim.ReleaseMatrix(exp_QKt);
    gpu_sim.ReleaseMatrix(softmax_result);
    // final_result will be released by rater.CommitAnswer

    /*********************  End of your code *********************/
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu