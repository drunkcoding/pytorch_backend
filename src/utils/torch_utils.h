#include <torch/script.h>

typedef torch::Device Device;

#define CPU_DEVICE torch::Device(torch::kCPU)
#define CUDA_DEVICE(index) torch::Device(torch::kCUDA, index)
#define DISK_DEVICE torch::Device(torch::kLazy)
#define DEFAULT_CUDA_DEVICE torch::Device(torch::kCUDA, 0)

typedef torch::jit::script::Module ScriptModule;
// typedef std::shared_ptr<Module> LibTorchModulePtr;
typedef ScriptModule* ScriptModulePtr;  // always use raw pointer, since we need
                                        // to manage the memory by ourselves