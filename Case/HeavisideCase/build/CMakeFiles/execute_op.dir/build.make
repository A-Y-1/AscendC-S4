# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/ypy/AscendC-S4/Case/HeavisideCase/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/ypy/AscendC-S4/Case/HeavisideCase/build

# Include any dependencies generated for this target.
include CMakeFiles/execute_op.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/execute_op.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/execute_op.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/execute_op.dir/flags.make

CMakeFiles/execute_op.dir/operator_desc.cpp.o: CMakeFiles/execute_op.dir/flags.make
CMakeFiles/execute_op.dir/operator_desc.cpp.o: /root/ypy/AscendC-S4/Case/HeavisideCase/src/operator_desc.cpp
CMakeFiles/execute_op.dir/operator_desc.cpp.o: CMakeFiles/execute_op.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/ypy/AscendC-S4/Case/HeavisideCase/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/execute_op.dir/operator_desc.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/execute_op.dir/operator_desc.cpp.o -MF CMakeFiles/execute_op.dir/operator_desc.cpp.o.d -o CMakeFiles/execute_op.dir/operator_desc.cpp.o -c /root/ypy/AscendC-S4/Case/HeavisideCase/src/operator_desc.cpp

CMakeFiles/execute_op.dir/operator_desc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/execute_op.dir/operator_desc.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/ypy/AscendC-S4/Case/HeavisideCase/src/operator_desc.cpp > CMakeFiles/execute_op.dir/operator_desc.cpp.i

CMakeFiles/execute_op.dir/operator_desc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/execute_op.dir/operator_desc.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/ypy/AscendC-S4/Case/HeavisideCase/src/operator_desc.cpp -o CMakeFiles/execute_op.dir/operator_desc.cpp.s

CMakeFiles/execute_op.dir/op_runner.cpp.o: CMakeFiles/execute_op.dir/flags.make
CMakeFiles/execute_op.dir/op_runner.cpp.o: /root/ypy/AscendC-S4/Case/HeavisideCase/src/op_runner.cpp
CMakeFiles/execute_op.dir/op_runner.cpp.o: CMakeFiles/execute_op.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/ypy/AscendC-S4/Case/HeavisideCase/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/execute_op.dir/op_runner.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/execute_op.dir/op_runner.cpp.o -MF CMakeFiles/execute_op.dir/op_runner.cpp.o.d -o CMakeFiles/execute_op.dir/op_runner.cpp.o -c /root/ypy/AscendC-S4/Case/HeavisideCase/src/op_runner.cpp

CMakeFiles/execute_op.dir/op_runner.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/execute_op.dir/op_runner.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/ypy/AscendC-S4/Case/HeavisideCase/src/op_runner.cpp > CMakeFiles/execute_op.dir/op_runner.cpp.i

CMakeFiles/execute_op.dir/op_runner.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/execute_op.dir/op_runner.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/ypy/AscendC-S4/Case/HeavisideCase/src/op_runner.cpp -o CMakeFiles/execute_op.dir/op_runner.cpp.s

CMakeFiles/execute_op.dir/main.cpp.o: CMakeFiles/execute_op.dir/flags.make
CMakeFiles/execute_op.dir/main.cpp.o: /root/ypy/AscendC-S4/Case/HeavisideCase/src/main.cpp
CMakeFiles/execute_op.dir/main.cpp.o: CMakeFiles/execute_op.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/ypy/AscendC-S4/Case/HeavisideCase/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/execute_op.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/execute_op.dir/main.cpp.o -MF CMakeFiles/execute_op.dir/main.cpp.o.d -o CMakeFiles/execute_op.dir/main.cpp.o -c /root/ypy/AscendC-S4/Case/HeavisideCase/src/main.cpp

CMakeFiles/execute_op.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/execute_op.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/ypy/AscendC-S4/Case/HeavisideCase/src/main.cpp > CMakeFiles/execute_op.dir/main.cpp.i

CMakeFiles/execute_op.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/execute_op.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/ypy/AscendC-S4/Case/HeavisideCase/src/main.cpp -o CMakeFiles/execute_op.dir/main.cpp.s

CMakeFiles/execute_op.dir/common.cpp.o: CMakeFiles/execute_op.dir/flags.make
CMakeFiles/execute_op.dir/common.cpp.o: /root/ypy/AscendC-S4/Case/HeavisideCase/src/common.cpp
CMakeFiles/execute_op.dir/common.cpp.o: CMakeFiles/execute_op.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/ypy/AscendC-S4/Case/HeavisideCase/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/execute_op.dir/common.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/execute_op.dir/common.cpp.o -MF CMakeFiles/execute_op.dir/common.cpp.o.d -o CMakeFiles/execute_op.dir/common.cpp.o -c /root/ypy/AscendC-S4/Case/HeavisideCase/src/common.cpp

CMakeFiles/execute_op.dir/common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/execute_op.dir/common.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/ypy/AscendC-S4/Case/HeavisideCase/src/common.cpp > CMakeFiles/execute_op.dir/common.cpp.i

CMakeFiles/execute_op.dir/common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/execute_op.dir/common.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/ypy/AscendC-S4/Case/HeavisideCase/src/common.cpp -o CMakeFiles/execute_op.dir/common.cpp.s

# Object files for target execute_op
execute_op_OBJECTS = \
"CMakeFiles/execute_op.dir/operator_desc.cpp.o" \
"CMakeFiles/execute_op.dir/op_runner.cpp.o" \
"CMakeFiles/execute_op.dir/main.cpp.o" \
"CMakeFiles/execute_op.dir/common.cpp.o"

# External object files for target execute_op
execute_op_EXTERNAL_OBJECTS =

/root/ypy/AscendC-S4/Case/HeavisideCase/output/execute_op: CMakeFiles/execute_op.dir/operator_desc.cpp.o
/root/ypy/AscendC-S4/Case/HeavisideCase/output/execute_op: CMakeFiles/execute_op.dir/op_runner.cpp.o
/root/ypy/AscendC-S4/Case/HeavisideCase/output/execute_op: CMakeFiles/execute_op.dir/main.cpp.o
/root/ypy/AscendC-S4/Case/HeavisideCase/output/execute_op: CMakeFiles/execute_op.dir/common.cpp.o
/root/ypy/AscendC-S4/Case/HeavisideCase/output/execute_op: CMakeFiles/execute_op.dir/build.make
/root/ypy/AscendC-S4/Case/HeavisideCase/output/execute_op: CMakeFiles/execute_op.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/ypy/AscendC-S4/Case/HeavisideCase/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable /root/ypy/AscendC-S4/Case/HeavisideCase/output/execute_op"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/execute_op.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/execute_op.dir/build: /root/ypy/AscendC-S4/Case/HeavisideCase/output/execute_op
.PHONY : CMakeFiles/execute_op.dir/build

CMakeFiles/execute_op.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/execute_op.dir/cmake_clean.cmake
.PHONY : CMakeFiles/execute_op.dir/clean

CMakeFiles/execute_op.dir/depend:
	cd /root/ypy/AscendC-S4/Case/HeavisideCase/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/ypy/AscendC-S4/Case/HeavisideCase/src /root/ypy/AscendC-S4/Case/HeavisideCase/src /root/ypy/AscendC-S4/Case/HeavisideCase/build /root/ypy/AscendC-S4/Case/HeavisideCase/build /root/ypy/AscendC-S4/Case/HeavisideCase/build/CMakeFiles/execute_op.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/execute_op.dir/depend

