#include <torch/csrc/distributed/c10d/ProcessGroupMPI_MOSE.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#ifdef USE_C10D_MPI

#include <iostream>
#include <map>

#include <c10/core/DeviceGuard.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <mpi.h>
#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h> // Needed for CUDA-aware check
#endif

#include "mose.h"

//

#define MPI_CHECK(cmd)                                                   \
  do {                                                                   \
    int mpiStatus = cmd;                                                 \
    if (mpiStatus != MPI_SUCCESS) {                                      \
      std::string err = "MPI error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) +                                     \
          ", with error code: " + std::to_string(mpiStatus);             \
      TORCH_CHECK(false, err);                                           \
    }                                                                    \
  } while (0)

//

namespace c10d {

namespace {

// Op mapping
std::map<ReduceOp::RedOpType, MPI_Op> mpiOp = {
    {ReduceOp::MIN, MPI_MIN},
    {ReduceOp::MAX, MPI_MAX},
    {ReduceOp::SUM, MPI_SUM},
    {ReduceOp::PRODUCT, MPI_PROD},
};
// Type mapping
std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
};

// Checking CUDA-aware MPI support, currently we only support CUDA aware
// MPI ops through Open MPI
bool cudaAwareMpiCheck() {
// Run time check
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (MPIX_Query_cuda_support() == 1) {
    return true;
  } else {
    return false;
  }
#else // !defined(MPIX_CUDA_AWARE_SUPPORT)
  return false;
#endif // MPIX_CUDA_AWARE_SUPPORT
}

// Checking the input tensor's validity
void checkSingleTensorHelper(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    TORCH_CHECK(false, "input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    TORCH_CHECK(false, "input tensor has to be dense");
  }
  if (tensor.is_cuda() && !cudaAwareMpiCheck()) {
    TORCH_CHECK(
        false,
        "CUDA tensor detected and the MPI used doesn't "
        "have CUDA-aware MPI support");
  }
}

void checkSingleTensor(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    TORCH_CHECK(
        false, "MPI process group does not support multi-GPU collectives");
  }
  checkSingleTensorHelper(tensors[0]);
}

void checkSameSizeAndType(
    const at::Tensor& t_in,
    const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    if ((tensor.numel() != t_in.numel()) ||
        (tensor.scalar_type() != t_in.scalar_type())) {
      TORCH_CHECK(false, "Tensors are not equal in size or data type");
    }
    checkSingleTensorHelper(tensor);
  }
}

} // namespace

//-------------------------------------------------
// [SECTION] Async MPI Work related
//-------------------------------------------------

ProcessGroupMPI_MOSE::AsyncWork::AsyncWork(
    MPI_Request request,
    std::vector<at::Tensor> outputTensors,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    : Work(-1, OpType::UNKNOWN, profilingTitle, inputTensors),
      outputTensors_(std::move(outputTensors)),
      request_(request) {
  memset(&status_, 0, sizeof(status_));
}

ProcessGroupMPI_MOSE::AsyncWork::~AsyncWork() {
  if (request_ != MPI_REQUEST_NULL) {
    std::cerr
        << "Attempted destruction of AsyncWork before work has completed, "
        << "terminating the program." << '\n';
    std::terminate();
  }
}

bool ProcessGroupMPI_MOSE::AsyncWork::isCompleted() {
  if (request_ == MPI_REQUEST_NULL) {
    return true;
  }

  int flag = 0;
  MPI_CHECK(MPI_Test(&request_, &flag, &status_));
  if (request_ != MPI_REQUEST_NULL) {
    return false;
  }

  // request_ == MPI_REQUEST_NULL; the work has completed
  // Populate exception if request was not successful
  if (status_.MPI_ERROR != MPI_SUCCESS) {
    populateException();
  }

  return true;
}

bool ProcessGroupMPI_MOSE::AsyncWork::isSuccess() const {
  if (request_ != MPI_REQUEST_NULL) {
    TORCH_CHECK(
        false,
        "Invalid call to AsyncWork::isSuccess before work has completed");
  }

  return status_.MPI_ERROR == MPI_SUCCESS;
}

int ProcessGroupMPI_MOSE::AsyncWork::sourceRank() const {
  return status_.MPI_SOURCE;
}

bool ProcessGroupMPI_MOSE::AsyncWork::wait(
    std::chrono::milliseconds /* unused */) {
  if (request_ == MPI_REQUEST_NULL) {
    // AsyncWork needs to manually call profiling end callbacks if they are set,
    // since it does not call ProcessGroup::finish().
    if (Work::recordFunctionEndCallback_) {
      Work::recordFunctionEndCallback_();
      Work::recordFunctionEndCallback_ = nullptr;
    }
    return true;
  }

  MPI_CHECK(MPI_Wait(&request_, &status_));
  auto ok = (status_.MPI_ERROR == MPI_SUCCESS);

  // AsyncWork needs to manually call profiling end callbacks if they are set,
  // since it does not call ProcessGroup::finish().
  if (Work::recordFunctionEndCallback_) {
    Work::recordFunctionEndCallback_();
    Work::recordFunctionEndCallback_ = nullptr;
  }

  if (!ok) {
    populateException();
    std::rethrow_exception(exception_);
  }
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(
        c10::intrusive_ptr<ProcessGroupMPI_MOSE::AsyncWork>::
            unsafe_reclaim_from_nonowning(this));
  }
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupMPI_MOSE::AsyncWork::abort(){TORCH_CHECK(
    false,
    "ProcessGroupMPI_MOSE::AsyncWork::abort not implemented.")}

std::vector<at::Tensor> ProcessGroupMPI_MOSE::AsyncWork::result() {
  return outputTensors_;
}

void ProcessGroupMPI_MOSE::AsyncWork::populateException() {
  std::array<char, MPI_MAX_ERROR_STRING> buf{};
  int len = buf.size();
  MPI_CHECK(MPI_Error_string(status_.MPI_ERROR, buf.data(), &len));
  exception_ =
      std::make_exception_ptr(std::runtime_error(std::string(buf.data(), len)));
}

// ************************************************
// ************************************************
//
// MPI process group implementation
//
// ************************************************
// ************************************************

void ProcessGroupMPI_MOSE::mpiExit() {
  MPI_CHECK(MPI_Finalize());
}

c10::intrusive_ptr<ProcessGroupMPI_MOSE> ProcessGroupMPI_MOSE::
    createProcessGroupMPI_MOSE(std::vector<int> ranks) {
  MPI_CHECK(MPI_Init(nullptr, nullptr));

  MPI_Comm groupComm = MPI_COMM_WORLD;
  int rank = -1;
  int size = -1;

  {
    // If no ranks are specified, assume we're creating the root group
    if (!ranks.empty()) {
      MPI_Group worldGroup{};
      MPI_Group ranksGroup{};
      MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &worldGroup));
      MPI_CHECK(
          MPI_Group_incl(worldGroup, ranks.size(), ranks.data(), &ranksGroup));
      // `MPI_Comm_create` can be flaky in certain cases.
      // See: https://github.com/pytorch/pytorch/issues/53899
      constexpr int kMaxNumRetries = 3;
      bool groupComm_updated = false;
      MPI_Barrier(MPI_COMM_WORLD);
      for (const auto i : c10::irange(kMaxNumRetries)) {
        (void)i;
        if (MPI_Comm_create(MPI_COMM_WORLD, ranksGroup, &groupComm)) {
          groupComm_updated = true;
          break;
        }
      }
      MPI_CHECK(groupComm_updated);
      MPI_CHECK(MPI_Group_free(&worldGroup));
      MPI_CHECK(MPI_Group_free(&ranksGroup));
    }

    // Fetch rank and world size for this group (MPI_COMM_WORLD or new)
    if (groupComm != MPI_COMM_NULL) {
      MPI_CHECK(MPI_Comm_rank(groupComm, &rank));
      MPI_CHECK(MPI_Comm_size(groupComm, &size));

      if (rank < 0 || size < 0) {
        TORCH_CHECK(false, "Failed to get the world_size / rank");
      }
    }
  }

  // If this process is not part of the group, we don't construct a
  // process group instance. This is in line with the semantics of the
  // other process group types.
  if (groupComm == MPI_COMM_NULL) {
    return c10::intrusive_ptr<ProcessGroupMPI_MOSE>();
  }

  return c10::make_intrusive<ProcessGroupMPI_MOSE>(rank, size, groupComm);
}

ProcessGroupMPI_MOSE::ProcessGroupMPI_MOSE(int rank, int size, MPI_Comm pgComm)
    : Backend(rank, size), stop_(false), pgComm_(pgComm) {
  if (pgComm_ == MPI_COMM_NULL) {
    TORCH_CHECK(false, "pgComm_ must not be MPI_COMM_NULL");
  }

  if (0 == rank) {
    fprintf(stdout, "\nNOTE:  Using custom MPI-MOSE backend implementation.\n");
  }

  init();
}

ProcessGroupMPI_MOSE::~ProcessGroupMPI_MOSE() {
  destroy();
}

void ProcessGroupMPI_MOSE::destroy() {
  stop_ = true;
  return;
}

void ProcessGroupMPI_MOSE::abort() {
  destroy();
  MPI_Abort(pgComm_, EXIT_FAILURE);
}

//

//----------------------------------------------------
// [SECTION]: Allreduce
//----------------------------------------------------

c10::intrusive_ptr<Work> ProcessGroupMPI_MOSE::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  // c
  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    c10::DeviceGuard guard(tensor.device());
    MPI_CHECK(MPI_Iallreduce(
        MPI_IN_PLACE,
        tensor.data_ptr(),
        tensor.numel(),
        mpiDatatype.at(tensor.scalar_type()),
        mpiOp.at(opts.reduceOp),
        pgComm_,
        &request));
  }

  auto work = c10::make_intrusive<AsyncWork>(
      request,
      std::vector<at::Tensor>(),
      "mpi_mose:allreduce",
      std::optional<std::vector<at::Tensor>>(tensors));
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupMPI_MOSE::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  // c
  checkSingleTensor(inputTensors);
  if (outputTensors.size() != 1) {
    TORCH_CHECK(
        false,
        "MPI process group only supports a single "
        "tensor op");
  }
  if (static_cast<size_t>(size_) != outputTensors[0].size()) {
    TORCH_CHECK(
        false,
        "All gather: number of output tensors should equal "
        "to the world size");
  }
  checkSameSizeAndType(inputTensors[0], outputTensors[0]);

  auto& input_tensor = inputTensors[0];
  auto& output_tensor = outputTensors[0];
  auto flat_out_vec = newLikeFlat(output_tensor);

  MPI_Request request = MPI_REQUEST_NULL;

  {
    c10::DeviceGuard guard(input_tensor.device());
    MPI_CHECK(MPI_Allgather(
        input_tensor.data_ptr(),
        input_tensor.numel(),
        mpiDatatype.at(input_tensor.scalar_type()),
        flat_out_vec.data_ptr(),
        input_tensor.numel(),
        mpiDatatype.at(input_tensor.scalar_type()),
        pgComm_));

    for (const auto i : c10::irange(output_tensor.size())) {
      output_tensor[i].copy_(flat_out_vec[static_cast<int64_t>(i)]);
    }
  }

  auto work = c10::make_intrusive<AsyncWork>(
      request, std::vector<at::Tensor>(), "mpi_mose:allgather", std::nullopt);
  return work;
}

// NOTE: this barrier() implementation actually respect MPI semantics,
//       not returning until everyone get here.
c10::intrusive_ptr<Work> ProcessGroupMPI_MOSE::barrier(
    const BarrierOptions& opts) {
  // NOTE: init to REQUEST_NULL, so that any further work->wait() return
  // immediately
  MPI_Request dummy = MPI_REQUEST_NULL;
  {
    MPI_CHECK(MPI_Barrier(pgComm_));
  }

  auto work = c10::make_intrusive<AsyncWork>(
      dummy, std::vector<at::Tensor>(), "mpi_mose:barrier", std::nullopt);
  return work;
}

} // namespace c10d

#endif // USE_C10D_MPI
