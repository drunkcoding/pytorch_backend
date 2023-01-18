#include "shm_utils.h"

int shm_fd = -1;
std::uint32_t* g_cnt_shm = nullptr;
std::mutex g_mutex_shm;

void
OpenShm()
{
  if (shm_fd == -1) {
    shm_fd = shm_open("expert_count", O_RDWR, 0666);
    if (shm_fd == -1) {
      assert(false);
    }
  }
}

void
MMapShm()
{
  if (g_cnt_shm == nullptr) {
    g_cnt_shm = (uint32_t*)mmap(
        NULL, sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (g_cnt_shm == MAP_FAILED) {
      assert(false);
    }
  }
}

std::uint32_t
GetCntFromShm()
{
  std::lock_guard<std::mutex> lock(g_mutex_shm);

  OpenShm();
  MMapShm();

  return *g_cnt_shm;
}

void
DecCntShm()
{
  std::lock_guard<std::mutex> lock(g_mutex_shm);
  OpenShm();
  MMapShm();

  if ((*g_cnt_shm) > 0) {
    (*g_cnt_shm)--;
  }
}