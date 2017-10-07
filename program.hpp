#pragma once

#include <string>

namespace mfl {
  namespace cl {

    class Program {
    public:
      Program(const std::string &buildString) :
          mBuildString(buildString) {};

      virtual ~Program() = default;

      const char *const buildString() const {
        return mBuildString.c_str();
      }

      virtual const char *const path() const = 0;

      virtual const char *const name() const = 0;

    private:
      const std::string mBuildString;
    };
  }
}