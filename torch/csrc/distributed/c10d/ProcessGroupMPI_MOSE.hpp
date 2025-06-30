#pragma once

#include <ATen/core/jit_type_base.h>
#include <c10/core/DeviceType.h>
#include <c10/util/intrusive_ptr.h>
#ifdef USE_C10D_MPI

#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <mpi.h>

namespace c10d {

constexpr const char* MPI_MOSE_BACKEND_NAME = "mpi_mose";

//

class TORCH_API ProcessGroupMPI_MOSE : public Backend {
 public:
  class AsyncWork : public Work {
   public:
    AsyncWork(
        MPI_Request request,
        std::vector<at::Tensor> outputTensors,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            std::nullopt);

    ~AsyncWork() override;

    struct Future : public at::ivalue::Future {
      explicit Future(const at::TypePtr& type, MPI_Request* request)
          : at::ivalue::Future(type, {c10::kCPU}), request_(request) {}
      ~Future() override;

      void wait() override;

     private:
      MPI_Request* request_ = nullptr;
    };

    bool isCompleted() override;
    bool isSuccess() const override;
    int sourceRank() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
    void abort() override;
    std::vector<at::Tensor> result() override;

    c10::intrusive_ptr<at::ivalue::Future> getFuture() override;

   protected:
    void populateException();

   private:
    c10::intrusive_ptr<Future> future_;
    const std::vector<at::Tensor> outputTensors_;
    MPI_Request request_;
    MPI_Status status_{};
  };

  // Constructor will spawn up the worker thread loop
  explicit ProcessGroupMPI_MOSE(int rank, int size, MPI_Comm pgComm);
  ~ProcessGroupMPI_MOSE() override;

  // Abort the MPI program, needs to be called when exception is detected
  void abort() override;

  const std::string getBackendName() const override {
    return std::string(MPI_MOSE_BACKEND_NAME);
  }

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  // Creating a new ProcessGroupMPI_MOSE, will initialize MPI if not initialized
  static c10::intrusive_ptr<ProcessGroupMPI_MOSE> createProcessGroupMPI_MOSE(
      std::vector<int> ranks = {});

 protected:
  // Helper function that is called by the destructor
  void destroy();

  bool stop_;

  // Global states
  static void mpiExit();

  MPI_Comm pgComm_;
};

} // namespace c10d

#endif // USE_C10D_MPI
