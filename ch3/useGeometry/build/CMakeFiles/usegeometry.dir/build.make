# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ellis/code/slam/ch3/useGeometry

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ellis/code/slam/ch3/useGeometry/build

# Include any dependencies generated for this target.
include CMakeFiles/usegeometry.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/usegeometry.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/usegeometry.dir/flags.make

CMakeFiles/usegeometry.dir/main.cpp.o: CMakeFiles/usegeometry.dir/flags.make
CMakeFiles/usegeometry.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ellis/code/slam/ch3/useGeometry/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/usegeometry.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/usegeometry.dir/main.cpp.o -c /home/ellis/code/slam/ch3/useGeometry/main.cpp

CMakeFiles/usegeometry.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/usegeometry.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ellis/code/slam/ch3/useGeometry/main.cpp > CMakeFiles/usegeometry.dir/main.cpp.i

CMakeFiles/usegeometry.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/usegeometry.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ellis/code/slam/ch3/useGeometry/main.cpp -o CMakeFiles/usegeometry.dir/main.cpp.s

CMakeFiles/usegeometry.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/usegeometry.dir/main.cpp.o.requires

CMakeFiles/usegeometry.dir/main.cpp.o.provides: CMakeFiles/usegeometry.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/usegeometry.dir/build.make CMakeFiles/usegeometry.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/usegeometry.dir/main.cpp.o.provides

CMakeFiles/usegeometry.dir/main.cpp.o.provides.build: CMakeFiles/usegeometry.dir/main.cpp.o


# Object files for target usegeometry
usegeometry_OBJECTS = \
"CMakeFiles/usegeometry.dir/main.cpp.o"

# External object files for target usegeometry
usegeometry_EXTERNAL_OBJECTS =

usegeometry: CMakeFiles/usegeometry.dir/main.cpp.o
usegeometry: CMakeFiles/usegeometry.dir/build.make
usegeometry: CMakeFiles/usegeometry.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ellis/code/slam/ch3/useGeometry/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable usegeometry"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/usegeometry.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/usegeometry.dir/build: usegeometry

.PHONY : CMakeFiles/usegeometry.dir/build

CMakeFiles/usegeometry.dir/requires: CMakeFiles/usegeometry.dir/main.cpp.o.requires

.PHONY : CMakeFiles/usegeometry.dir/requires

CMakeFiles/usegeometry.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/usegeometry.dir/cmake_clean.cmake
.PHONY : CMakeFiles/usegeometry.dir/clean

CMakeFiles/usegeometry.dir/depend:
	cd /home/ellis/code/slam/ch3/useGeometry/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ellis/code/slam/ch3/useGeometry /home/ellis/code/slam/ch3/useGeometry /home/ellis/code/slam/ch3/useGeometry/build /home/ellis/code/slam/ch3/useGeometry/build /home/ellis/code/slam/ch3/useGeometry/build/CMakeFiles/usegeometry.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/usegeometry.dir/depend

