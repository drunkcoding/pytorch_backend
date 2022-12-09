#include "data_utils.h"

std::string
GetRequestId(TRITONBACKEND_Request* request)
{
  const char* request_id = nullptr;
  LOG_IF_ERROR(
      TRITONBACKEND_RequestCorrelationIdString(request, &request_id),
      "unable to retrieve request ID string");
  if ((request_id == nullptr) || (request_id[0] == '\0')) {
    request_id = "<id_unknown>";
  }
  return std::string(request_id);
}

std::uint64_t
GetCorrelationId(TRITONBACKEND_Request* request)
{
  std::uint64_t correlation_id;
  LOG_IF_ERROR(
      TRITONBACKEND_RequestCorrelationId(request, &correlation_id),
      "unable to retrieve correlation ID string");
  // if ((correlation_id == nullptr) || (correlation_id[0] == '\0')) {
  //   correlation_id = "<id_unknown>";
  //   static_assert(
  //       sizeof(CorrelationID) == sizeof(uint64_t),
  //       "CorrelationID must be 64-bit");
  // }
  return correlation_id;
}