#pragma once

#include <fcntl.h> /* For O_* constants */
#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <unistd.h>

#include <cassert>
#include <memory>
#include <mutex>

extern std::mutex g_mutex_shm;
extern int shm_fd;
extern std::uint32_t* g_cnt_shm;

void OpenShm();
void MMapShm();

std::uint32_t GetCntFromShm();
void DecCntShm();