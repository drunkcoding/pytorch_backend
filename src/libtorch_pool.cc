#include "libtorch_pool.h"
#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace pytorch {

ModelMetaPtr
LibTorchPool::RegisterModule(
    const std::string& model_path, const std::string& model_name,
    const std::uint64_t model_version)
{
  std::size_t model_id = MakeID(model_name, model_version);
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = registered_models_.find(model_id);
  if (it != registered_models_.end()) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("Model ") + model_name + " is already registered on " +
         std::to_string(it->second))
            .c_str());
    return *cache_[it->second].Get(model_id);  // already registered
  }

  // register to CPU memory first
  auto model = std::make_shared<ModelMeta>(
      model_path, model_name, model_version);
  cache_[-1].Put(model_id, model);
  registered_models_.insert({model_id, -1});

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE, (std::string("Model ") + model_name +
                                 " is registered on " + std::to_string(-1))
                                    .c_str());

  return model;  // success
}

// TRITONSERVER_Error*
// LibTorchPool::DeregisterModule(const std::string& model_name)
// {
//   std::size_t hash = std::hash<std::string>{}(model_name);
//   std::lock_guard<std::mutex> lock(mutex_);
//   auto it = modules_.find(hash);
//   if (it != modules_.end()) {
//     modules_.erase(it);
//   } else {
//     return TRITONSERVER_ErrorNew(
//         TRITONSERVER_ERROR_INTERNAL, "Model not registered");
//   }

//   return nullptr;  // success
// }

// TRITONSERVER_Error*
// LibTorchPool::FetchModule(
//     const std::string& model_name, const std::uint64_t model_version,
//     const torch::Device device)
// {
//   std::size_t model_id = MakeID(model_name, model_version);
//   std::lock_guard<std::mutex> lock(mutex_);

//   if (registered_models_.find(model_id) == registered_models_.end()) {
//     return TRITONSERVER_ErrorNew(
//         TRITONSERVER_ERROR_INTERNAL, "Model not registered");
//   }

//   ModelMetaPtr* model;
//   if (device.is_cuda()) {
//     for (auto& [gpu_id, cache] : cache_) {
//       model = cache.Get(model_id);
//       if (model != nullptr) {
//         break;
//       }
//     }

//     if (device.index() != model->get()->device().index()) {
//       // move to the target device
//       model = model->ReloadToDevice(device.index());
//       cache_[device.index()].Put(model_id, *model);
//       // remove the model from the original device
//       cache_[model->get()->device().index()].Remove(model_id);
//     }
//   }

//   return nullptr;  // success
// }

// TRITONSERVER_Error*
// LibTorchPool::ReleaseModule(
//     const std::string& model_name, const torch::Device device)
// {
//   std::size_t hash = std::hash<std::string>{}(model_name);
//   std::lock_guard<std::mutex> lock(mutex_);
//   auto it = modules_.find(hash);
//   if (it != modules_.end()) {
//     if (GetTorchDevice(device) == TorchDevice::DEVICE_CPU &&
//         it->second->model_device == TorchDevice::DEVICE_GPU) {
//       it->second->checkpoint->to(torch::kCPU);
//       it->second->model_device = TorchDevice::DEVICE_CPU;
//     } else if (device == TorchDevice::DEVICE_DISK) {
//       it->second->checkpoint = nullptr;
//       it->second->model_device = TorchDevice::DEVICE_DISK;
//     } else {
//       return TRITONSERVER_ErrorNew(
//           TRITONSERVER_ERROR_INTERNAL, "Invalid device");
//     }
//   } else {
//     return TRITONSERVER_ErrorNew(
//         TRITONSERVER_ERROR_INTERNAL, "Model not registered");
//   }

//   return nullptr;  // success
// }

// LibTorchModulePtr
// LibTorchPool::GetModule(
//     const std::string& model_name, const std::uint64_t model_version)
// {
//   std::size_t model_id = MakeID(model_name, model_version);
//   std::lock_guard<std::mutex> lock(mutex_);
//   auto it = modules_.find(hash);
//   if (it != modules_.end()) {
//     return it->second->checkpoint;
//   } else {
//     return nullptr;
//   }
// }

}}}  // namespace triton::backend::pytorch