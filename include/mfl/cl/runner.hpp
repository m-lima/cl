#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <mfl/out.hpp>
#include <mfl/string.hpp>
#include <mfl/exception.hpp>

#include "program.hpp"

namespace mfl {
  namespace cl {

    class Runner {
    public:

      Runner(cl_device_type type,
             bool verbose = false,
             const std::vector<const char *> & requirements
             = std::vector<const char *>(0));

      void loadProgram(const Program & program, bool verbose = false);

      void releaseProgram(const std::string & name);

      std::vector<::cl::CommandQueue> commandQueues(std::size_t deviceCount);

      void releaseQueues();

      ::cl::Kernel makeKernel(const std::string & program,
                              const std::string & kernelName,
                              bool verbose = false);

      template<typename ... T>
      ::cl::make_kernel<T...> makeKernelFunctor(const std::string & program,
                                                const std::string & kernelName) {
        auto builtProgram = mPrograms.find(program);
        if (builtProgram == mPrograms.end()) {
          throw mfl::Exception::build("No program named {} has been loaded yet",
                                      program);
        }

        try {
          return ::cl::make_kernel<T...>(builtProgram->second, kernelName);
        } catch (::cl::Error & err) {
          throw mfl::Exception::build("OpenCL error: {} ({} : {})",
                                      err.what(),
                                      err.err(),
                                      getErrorString(err.err()));
        }
      }

      template<typename ... Args>
      const ::cl::Buffer & createBuffer(const std::string & name,
                                        const Args & ... args) {
        if (mBuffers.find(name) != mBuffers.end()) {
          throw mfl::Exception::build("Trying to create a buffer with an"
                                          "existing name");
        }

        try {
          return (mBuffers.emplace(name, ::cl::Buffer(mContext, args...))
              .first)->second;
        } catch (::cl::Error & err) {
          throw mfl::Exception::build("OpenCL error: {} ({} : {})",
                                      err.what(),
                                      err.err(),
                                      getErrorString(err.err()));
        }
      }

      const ::cl::Buffer & getBuffer(const std::string & name) const {
        return mBuffers.at(name);
      }

      void releaseBuffer(const std::string & name);

      size_t totalMemory() const {
        return mTotalMemory;
      }

      size_t bufferMemory() const {
        return mBufferMemory;
      }

      const ::cl::Context & context() const {
        return mContext;
      }

      operator const ::cl::Context &() const {
        return mContext;
      }

      static void printFull() {
        mfl::out::println("Listing all platforms and devices..");
        try {
          std::vector<::cl::Platform> platforms;
          ::cl::Platform::get(&platforms);
          std::string name;

          if (platforms.empty()) {
            throw mfl::Exception::build("OpenCL platforms not found.");
          }

          ::cl::Context context;
          std::vector<::cl::Device> devices;
          for (auto platform = platforms.begin();
               platform != platforms.end();
               platform++) {
            name = platform->getInfo<CL_PLATFORM_NAME>();
            mfl::string::trimInPlace(name);
            mfl::out::println("{}", name);
            std::vector<::cl::Device> platformDevices;

            try {
              platform->getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);

              for (auto device = platformDevices.begin();
                   device != platformDevices.end();
                   device++) {
                name = device->getInfo<CL_DEVICE_NAME>();
                mfl::string::trimInPlace(name);
                mfl::out::println(" * {}", name);
              }
            } catch (...) {
            }
          }
        } catch (const ::cl::Error & err) {
          throw mfl::Exception::build("OpenCL error: {} ({} : {})",
                                      err.what(),
                                      err.err(),
                                      getErrorString(err.err()));
        }
      }

      static const char * getErrorString(cl_int error) {
        switch (error) {
          // run-time and JIT compiler errors
          case 0:
            return "CL_SUCCESS";
          case -1:
            return "CL_DEVICE_NOT_FOUND";
          case -2:
            return "CL_DEVICE_NOT_AVAILABLE";
          case -3:
            return "CL_COMPILER_NOT_AVAILABLE";
          case -4:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
          case -5:
            return "CL_OUT_OF_RESOURCES";
          case -6:
            return "CL_OUT_OF_HOST_MEMORY";
          case -7:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
          case -8:
            return "CL_MEM_COPY_OVERLAP";
          case -9:
            return "CL_IMAGE_FORMAT_MISMATCH";
          case -10:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
          case -11:
            return "CL_BUILD_PROGRAM_FAILURE";
          case -12:
            return "CL_MAP_FAILURE";
          case -13:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
          case -14:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
          case -15:
            return "CL_COMPILE_PROGRAM_FAILURE";
          case -16:
            return "CL_LINKER_NOT_AVAILABLE";
          case -17:
            return "CL_LINK_PROGRAM_FAILURE";
          case -18:
            return "CL_DEVICE_PARTITION_FAILED";
          case -19:
            return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

            // compile-time errors
          case -30:
            return "CL_INVALID_VALUE";
          case -31:
            return "CL_INVALID_DEVICE_TYPE";
          case -32:
            return "CL_INVALID_PLATFORM";
          case -33:
            return "CL_INVALID_DEVICE";
          case -34:
            return "CL_INVALID_CONTEXT";
          case -35:
            return "CL_INVALID_QUEUE_PROPERTIES";
          case -36:
            return "CL_INVALID_COMMAND_QUEUE";
          case -37:
            return "CL_INVALID_HOST_PTR";
          case -38:
            return "CL_INVALID_MEM_OBJECT";
          case -39:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
          case -40:
            return "CL_INVALID_IMAGE_SIZE";
          case -41:
            return "CL_INVALID_SAMPLER";
          case -42:
            return "CL_INVALID_BINARY";
          case -43:
            return "CL_INVALID_BUILD_OPTIONS";
          case -44:
            return "CL_INVALID_PROGRAM";
          case -45:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
          case -46:
            return "CL_INVALID_KERNEL_NAME";
          case -47:
            return "CL_INVALID_KERNEL_DEFINITION";
          case -48:
            return "CL_INVALID_KERNEL";
          case -49:
            return "CL_INVALID_ARG_INDEX";
          case -50:
            return "CL_INVALID_ARG_VALUE";
          case -51:
            return "CL_INVALID_ARG_SIZE";
          case -52:
            return "CL_INVALID_KERNEL_ARGS";
          case -53:
            return "CL_INVALID_WORK_DIMENSION";
          case -54:
            return "CL_INVALID_WORK_GROUP_SIZE";
          case -55:
            return "CL_INVALID_WORK_ITEM_SIZE";
          case -56:
            return "CL_INVALID_GLOBAL_OFFSET";
          case -57:
            return "CL_INVALID_EVENT_WAIT_LIST";
          case -58:
            return "CL_INVALID_EVENT";
          case -59:
            return "CL_INVALID_OPERATION";
          case -60:
            return "CL_INVALID_GL_OBJECT";
          case -61:
            return "CL_INVALID_BUFFER_SIZE";
          case -62:
            return "CL_INVALID_MIP_LEVEL";
          case -63:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
          case -64:
            return "CL_INVALID_PROPERTY";
          case -65:
            return "CL_INVALID_IMAGE_DESCRIPTOR";
          case -66:
            return "CL_INVALID_COMPILER_OPTIONS";
          case -67:
            return "CL_INVALID_LINKER_OPTIONS";
          case -68:
            return "CL_INVALID_DEVICE_PARTITION_COUNT";

            // extension errors
          case -1000:
            return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
          case -1001:
            return "CL_PLATFORM_NOT_FOUND_KHR";
          case -1002:
            return "CL_INVALID_D3D10_DEVICE_KHR";
          case -1003:
            return "CL_INVALID_D3D10_RESOURCE_KHR";
          case -1004:
            return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
          case -1005:
            return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
          default:
            return "Unknown OpenCL error";
        }
      }

    private:
      ::cl::Context mContext;
      std::vector<::cl::Device> mDevices;
      std::unordered_map<std::string, ::cl::Program> mPrograms;
      std::vector<::cl::CommandQueue> mCommands;
      std::unordered_map<std::string, ::cl::Buffer> mBuffers;

      size_t mTotalMemory;
      size_t mBufferMemory;
    };
  }
}
