#pragma once

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "../exception.hpp"
#include "../string.hpp"
#include "../out.hpp"

namespace mfl {
  namespace cl {
    namespace util {
      /////////////////////////////////////
      // OpenCL Helper
      inline std::vector<cl::Device> getGPUDevices() {
        std::vector<cl::Device> devices;

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) {
          mfl::out::println(stderr, "OpenCL platforms not found");
          return devices;
        }

        std::vector<cl::Device> platformDevices;
        for (auto &platform : platforms) {
          try {
            platform.getDevices(CL_DEVICE_TYPE_GPU, &platformDevices);
            for (auto &device : platformDevices) {
              devices.push_back(device);
              break;
            }
            if (!devices.empty()) {
              break;
            }
          } catch (...) {
            devices.clear();
          }
        }

        if (devices.empty()) {
          mfl::out::println(stderr, "No devices found");
        }

        return devices;
      }

      inline void printFull() {
        mfl::out::println("Listing all platforms and devices..");
        try {
          std::vector<cl::Platform> platforms;
          cl::Platform::get(&platforms);
          std::string name;

          if (platforms.empty()) {
            throw mfl::Exception::build("OpenCL platforms not found.");
          }

          cl::Context context;
          std::vector<cl::Device> devices;
          for (auto platform = platforms.begin();
               platform != platforms.end();
               platform++) {
            name = platform->getInfo<CL_PLATFORM_NAME>();
            mfl::string::trimInPlace(name);
            mfl::out::println("{}", name);
            std::vector<cl::Device> platformDevices;

            try {
              platform->getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);

              for (auto device = platformDevices.begin();
                   device != platformDevices.end();
                   device++) {
                name = device->getInfo<CL_DEVICE_NAME>();
                mfl::string::trimInPlace(name);
                mfl::out::println(" {} {}",
                                  ((device + 1) != platformDevices.end() ? "├" : "└"),
                                  name);
              }
            } catch (...) {
            }
          }
        } catch (const cl::Error &err) {
          throw mfl::Exception::build("OpenCL error: {} ({})",
                                      err.what(),
                                      err.err());
        }
      }

      inline void printLongDeviceInfo(const cl::Device &device) {
        mfl::out::println(
            "=========================|\n"
                "Name:                     {}\n"
                "OpenCL version:           {}\n"
                "Global memory size:       {}MB\n"
                "Local memory size:        {}KB\n"
                "Max constant buffer size: {}KB\n"
                "Max clock frequency:      {}Hz\n"
                "Max compute units:        {}\n"
                "Max work group size:      {}\n"
                "Extensions:               {}\n"
                "=========================|\n",
            device.getInfo<CL_DEVICE_NAME>(),
            device.getInfo<CL_DRIVER_VERSION>(),
            device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(),
            device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(),
            device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>(),
            device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>(),
            device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(),
            device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(),
            device.getInfo<CL_DEVICE_EXTENSIONS>()
        );
      }

    }
  }
}
