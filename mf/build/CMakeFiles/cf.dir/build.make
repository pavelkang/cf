# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.4

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.4.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.4.3/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/kai/Documents/rec/mf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/kai/Documents/rec/mf/build

# Include any dependencies generated for this target.
include CMakeFiles/cf.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cf.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cf.dir/flags.make

CMakeFiles/cf.dir/main.cpp.o: CMakeFiles/cf.dir/flags.make
CMakeFiles/cf.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/kai/Documents/rec/mf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cf.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cf.dir/main.cpp.o -c /Users/kai/Documents/rec/mf/main.cpp

CMakeFiles/cf.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cf.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/kai/Documents/rec/mf/main.cpp > CMakeFiles/cf.dir/main.cpp.i

CMakeFiles/cf.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cf.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/kai/Documents/rec/mf/main.cpp -o CMakeFiles/cf.dir/main.cpp.s

CMakeFiles/cf.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/cf.dir/main.cpp.o.requires

CMakeFiles/cf.dir/main.cpp.o.provides: CMakeFiles/cf.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/cf.dir/build.make CMakeFiles/cf.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/cf.dir/main.cpp.o.provides

CMakeFiles/cf.dir/main.cpp.o.provides.build: CMakeFiles/cf.dir/main.cpp.o


# Object files for target cf
cf_OBJECTS = \
"CMakeFiles/cf.dir/main.cpp.o"

# External object files for target cf
cf_EXTERNAL_OBJECTS =

cf: CMakeFiles/cf.dir/main.cpp.o
cf: CMakeFiles/cf.dir/build.make
cf: CMakeFiles/cf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/kai/Documents/rec/mf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cf"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cf.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cf.dir/build: cf

.PHONY : CMakeFiles/cf.dir/build

CMakeFiles/cf.dir/requires: CMakeFiles/cf.dir/main.cpp.o.requires

.PHONY : CMakeFiles/cf.dir/requires

CMakeFiles/cf.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cf.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cf.dir/clean

CMakeFiles/cf.dir/depend:
	cd /Users/kai/Documents/rec/mf/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/kai/Documents/rec/mf /Users/kai/Documents/rec/mf /Users/kai/Documents/rec/mf/build /Users/kai/Documents/rec/mf/build /Users/kai/Documents/rec/mf/build/CMakeFiles/cf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cf.dir/depend

