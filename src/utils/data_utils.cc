#include "data_utils.h"

std::uint64_t
GetRequestID(TRITONBACKEND_Request* request)
{
  const char* request_id = nullptr;
  LOG_IF_ERROR(
      TRITONBACKEND_RequestId(request, &request_id),
      "unable to retrieve request ID string");
  if ((request_id == nullptr) || (request_id[0] == '\0')) {
    request_id = "<id_unknown>";
  }
  return std::hash<std::string>{}(std::string(request_id));
}

std::uint64_t
GetCorrelationID(TRITONBACKEND_Request* request)
{
  std::uint64_t correlation_id;
  LOG_IF_ERROR(
      TRITONBACKEND_RequestCorrelationId(request, &correlation_id),
      "unable to retrieve correlation ID string");
  return correlation_id;
}