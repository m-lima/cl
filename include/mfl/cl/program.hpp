#pragma once

#include <string>

namespace mfl {
  namespace cl {

    class Program {
    public:
      Program(const std::string & buildString) :
          mBuildString(buildString) {};

      virtual ~Program() = default;

      const char * buildString() const {
        return mBuildString.c_str();
      }

      virtual const char * path() const = 0;

      virtual std::string getSource() const {
        return "";
      };

      virtual const char * name() const = 0;

    private:
      const std::string mBuildString;
    };
  }
}