// This file first loads a torchscript model using torch::jit::load() and then

// exports it to a ptf archive using torch::model::save().

#include <torch/script.h>  // One-stop header.

#include <chrono>
#include <iostream>

#include "gds_load.h"


int
load(
    const std::string& tensor_filename, const std::string& meta_filename,

    const torch::Device& device, torch::jit::script::Module& module)
{
  GDSLoader loader;

  // loader.cudaInit();

  try {
    void* tensor_pool = nullptr;

    // GDSLoader loader;

    int ret = loader.load(tensor_filename.c_str(), 0, &tensor_pool);


    // load asynchrously

    // std::future<ssize_t> read_bytes;

    // int ret = loader.loadAsync(tensor_filename.c_str(), 0, &tensor_pool,

    // read_bytes); int ret = loader.loadAsyncInited(tensor_filename.c_str(), 0,

    // &tensor_pool, read_bytes); // with cuda inited

    if (ret != 0) {
      std::cerr << "error loading the tensor file " << tensor_filename

                << std::endl;

      return -1;
    }

    module = torch::jit::fastLoad(meta_filename, device, tensor_pool);

    // start timer

    // auto start2 = std::chrono::high_resolution_clock::now();

    // read_bytes.wait();

    // auto end2 = std::chrono::high_resolution_clock::now();

    // auto duration2 =

    // std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);

    // std::cout << "read_bytes: " << read_bytes.get() << " elapsed: " <<

    // duration2.count() << " ms" << std::endl;
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model " << meta_filename << " " << e.msg()

              << std::endl;

    return -1;
  }


  return 0;
}