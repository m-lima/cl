#include "include/mfl/cl/runner.hpp"

#include <fstream>
#include <numeric>

#include <mfl/exception.hpp>

namespace mfl {
  namespace cl {

    Runner::Runner(cl_device_type type,
                   const std::vector<const char *> &requirements) {
      try {
        std::vector<::cl::Platform> platforms;
        ::cl::Platform::get(&platforms);

        if (platforms.empty()) {
          throw mfl::Exception::build("OpenCL platforms not found");
        }

        mfl::out::println("Detecting best platform..");
        std::vector<::cl::Device> platformDevices;
        size_t bestCount = 0;
        int bestIndex = 0;
        for (std::size_t i = 0; i < platforms.size(); ++i) {
          platforms[i].getDevices(type, &platformDevices);
          auto deviceCount = platformDevices.size();

          if (!requirements.empty()) {
            for (auto device : platformDevices) {
              std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
              for (const char *requirement : requirements) {
                if (extensions.find(requirement) == std::string::npos) {
                  deviceCount--;
                  break;
                }
              }
            }
          }

          if (deviceCount > bestCount) {
            bestCount = deviceCount;
            bestIndex = i;
          }
        }

        if (bestCount == 0) {
          throw mfl::Exception::build("No compatible OpenCL device found");
        }

        mfl::out::println("Chose {} with {} compatible device{}",
                          platforms[bestIndex].getInfo<CL_PLATFORM_NAME>(),
                          bestCount,
                          bestCount > 1 ? 's' : ' ');

        mTotalMemory = SIZE_MAX;
        mBufferMemory = SIZE_MAX;
        mDevices.reserve(bestCount);
        if (requirements.empty()) {
          platforms[bestIndex].getDevices(type, &mDevices);

          for (auto device : mDevices) {
            auto memory = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
            if (memory < mTotalMemory) {
              mTotalMemory = memory;
            }

            memory = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
            if (memory < mBufferMemory) {
              mBufferMemory = memory;
            }
          }
        } else {
          platforms[bestIndex].getDevices(type, &platformDevices);
          for (auto device : platformDevices) {
            std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
            bool compatible = true;
            for (const char *requirement : requirements) {
              if (extensions.find(requirement) == std::string::npos) {
                compatible = false;
                break;
              }
            }
            if (compatible) {
              mDevices.push_back(device);
              auto memory = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
              if (memory < mTotalMemory) {
                mTotalMemory = memory;
              }

              memory = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
              if (memory < mBufferMemory) {
                mBufferMemory = memory;
              }
            }
          }
        }

        mContext = ::cl::Context(mDevices);
      } catch (::cl::Error &err) {
        throw mfl::Exception::build("OpenCL error: {} ({} : {})",
                                    err.what(),
                                    err.err(),
                                    getErrorString(err.err()));
      }

      mfl::out::println();
    }

    void Runner::loadProgram(const Program &program) {
      if (mDevices.empty()) {
        throw mfl::Exception::build("Trying to load program without devices");
      }

      if (mPrograms.find(program.name()) != mPrograms.end()) {
        throw mfl::Exception::build("Trying to create a program with an"
                                        "existing name");
      }

      try {
        auto clProgram = ::cl::Program(mContext, program.getSource());

        mfl::out::println("Build log for {} ({})", program.name(), program.path());

        for (auto device : mDevices) {
          auto buildInfo = clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
          if (buildInfo.size() > 1) {
            mfl::out::println("== Device {}:\n"
                                  "{}\n"
                                  "=========",
                              device.getInfo<CL_DEVICE_NAME>(),
                              buildInfo);
          }
        }
        mfl::out::println();

        try {
          clProgram.build(mDevices, program.buildString());

#if defined(DEBUG) || defined(_DEBUG)
          auto assembly = clProgram.getInfo<CL_PROGRAM_BINARIES>();
          mfl::out::println("== Assembly:");
          for (auto line : assembly) {
            mfl::out::println("{}", line);
          }
#endif

          mPrograms[program.name()] = std::move(clProgram);
        } catch (::cl::Error &err) {
          if (err.err() == CL_INVALID_PROGRAM_EXECUTABLE
              || err.err() == CL_BUILD_ERROR
              || err.err() == CL_BUILD_PROGRAM_FAILURE) {
            for (auto device : mDevices) {
              mfl::out::println(stderr,
                                "Build failure\n{}",
                                clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
            }
          }
          throw mfl::Exception::build("OpenCL error: {} ({} : {})",
                                      err.what(),
                                      err.err(),
                                      getErrorString(err.err()));
        }
      } catch (::cl::Error &err) {
        throw mfl::Exception::build("OpenCL error: {} ({} : {})",
                                    err.what(),
                                    err.err(),
                                    getErrorString(err.err()));
      }
    }

    void Runner::releaseProgram(const std::string &name) {
      mPrograms.erase(name);
    }

    std::vector<::cl::CommandQueue> Runner::commandQueues(std::size_t deviceCount) {
      if (deviceCount <= 0) {
        return std::vector<::cl::CommandQueue>(0);
      }

      if (mCommands.size() < deviceCount) {
        mCommands.reserve(deviceCount);

        try {
          for (size_t i = mCommands.size(); i < deviceCount; ++i) {
            mCommands.emplace_back(mContext, mDevices[i]);
          }
        } catch (::cl::Error &err) {
          throw mfl::Exception::build("OpenCL error: {} ({} : {})",
                                      err.what(),
                                      err.err(),
                                      getErrorString(err.err()));
        }

        if (mCommands.size() < deviceCount) {
          throw mfl::Exception::build("Could not create command queues");
        }
      }

      return std::vector<::cl::CommandQueue>(mCommands.cbegin(),
                                           mCommands.cbegin() + deviceCount);
    }

    void Runner::releaseQueues() {
      mCommands = std::vector<::cl::CommandQueue>(0);
    }

    ::cl::Kernel Runner::makeKernel(const std::string &program,
                                  const std::string &kernelName) {

      auto builtProgram = mPrograms.find(program);
      if (builtProgram == mPrograms.end()) {
        throw mfl::Exception::build("No program named {} has been loaded yet",
                                    program);
      }

      ::cl::Kernel kernel(builtProgram->second, kernelName.c_str());

      for (auto device : mDevices) {
        mfl::out::println("Kernel info for {}", device.getInfo<CL_DEVICE_NAME>());
        auto compileWorkGroupSize =
            kernel.getWorkGroupInfo<CL_KERNEL_COMPILE_WORK_GROUP_SIZE>(device);
        mfl::out::println(" * Compile work group size:        {}, {}, {}",
                          compileWorkGroupSize[0],
                          compileWorkGroupSize[1],
                          compileWorkGroupSize[2]);
        size_t globalSize[3];
        clGetKernelWorkGroupInfo(kernel(),
                                 device(),
                                 CL_KERNEL_GLOBAL_WORK_SIZE,
                                 sizeof(globalSize),
                                 globalSize,
                                 0);
        mfl::out::println(" * Global work size:               {}, {}, {}",
                          globalSize[0],
                          globalSize[1],
                          globalSize[2]);
        mfl::out::println(" * Local memory size:              {}B",
                          kernel.getWorkGroupInfo
                              <CL_KERNEL_LOCAL_MEM_SIZE>(device));
        mfl::out::println(" * Preferred group size multiple:  {}",
                          kernel.getWorkGroupInfo
                              <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device));
        mfl::out::println(" * Private memory size:            {}B",
                          kernel.getWorkGroupInfo
                              <CL_KERNEL_PRIVATE_MEM_SIZE>(device));
        mfl::out::println(" * Work group size:                {}",
                          kernel.getWorkGroupInfo
                              <CL_KERNEL_WORK_GROUP_SIZE>(device));
        mfl::out::println();
      }

      return kernel;
    }

    void Runner::releaseBuffer(const std::string &name) {
      mBuffers.erase(name);
    }

  }
}
