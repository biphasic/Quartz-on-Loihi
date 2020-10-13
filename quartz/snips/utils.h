#pragma once
#include "nxsdk.h"

#define LOGGING 1

#if LOGGING
#define LOG(f_, ...) printf((f_), ##__VA_ARGS__)
#else
#define LOG(f_, ...) do { } while(0)
#endif

void logicalToPhysicalCoreId(int logicalId, CoreId *physicalId);