cmake_minimum_required(VERSION 3.22)
project(pycppdlr VERSION 0.1.0 LANGUAGES C CXX)

# cppdlr 1.3.0: 5f9adf91d7620ce145ead8e4e906151493038af3
# Pinned commits (not moving branches):
# itertools: 2bb4da2b56ab49b3cd7b5f27cfb88c5320fe8524
# mpi:       c745cd5049510c0fb981042dde6b34b6d8c346ab
# h5:        fca9584b2390745cad33fc513094d35921ccdbe2 (unstable, 2026-06-24)
# nda:       826bca37d014b82beaf15d6b89d3ec20cd9b82cb (unstable, 2026-06-24)
# nda 1.3.0 lacks nda::linalg::{matmul,matvecmul} required by cppdlr 1.3.0.
# h5 1.3.0 lacks the shape-pointer array API required by the fallback nda.
set(Build_Tests OFF CACHE BOOL "" FORCE)
set(Build_Documentation OFF CACHE BOOL "" FORCE)
set(Build_Benchs OFF CACHE BOOL "" FORCE)
set(PythonSupport OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(HDF5Support ON CACHE BOOL "" FORCE)
set(MPISupport ON CACHE BOOL "" FORCE)
set(OpenMPSupport ON CACHE BOOL "" FORCE)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
# Public rf inputs must raise exceptions if upstream node selection loses rank.
add_compile_definitions(NDA_ENFORCE_BOUNDCHECK)

# shortcut: skip test-only deps for this extension; revisit if enabling upstream tests/install exports.
set_property(GLOBAL PROPERTY GTest_FOUND TRUE)
set_property(GLOBAL PROPERTY fmt_FOUND TRUE)

include(FetchContent)
foreach(dep IN ITEMS itertools mpi h5 nda)
  if(dep STREQUAL "itertools")
    set(revision 2bb4da2b56ab49b3cd7b5f27cfb88c5320fe8524)
  elseif(dep STREQUAL "mpi")
    set(revision c745cd5049510c0fb981042dde6b34b6d8c346ab)
  elseif(dep STREQUAL "h5")
    set(revision fca9584b2390745cad33fc513094d35921ccdbe2)
  else()
    set(revision 826bca37d014b82beaf15d6b89d3ec20cd9b82cb)
  endif()
  FetchContent_Declare(${dep} GIT_REPOSITORY https://github.com/TRIQS/${dep}.git GIT_TAG ${revision})
  FetchContent_MakeAvailable(${dep})
  set_property(GLOBAL PROPERTY ${dep}_FOUND TRUE)
endforeach()

set(cppdlr_source "${CMAKE_CURRENT_SOURCE_DIR}/../cppdlr")
if(NOT EXISTS "${cppdlr_source}/CMakeLists.txt")
  message(FATAL_ERROR "Checkout build requires the sibling cppdlr submodule; initialize it first. SDists are unsupported.")
endif()
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${cppdlr_source}"
  OUTPUT_VARIABLE cppdlr_sha OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT cppdlr_sha STREQUAL "5f9adf91d7620ce145ead8e4e906151493038af3")
  message(FATAL_ERROR "Expected cppdlr 1.3.0 commit 5f9adf9, found ${cppdlr_sha}")
endif()
add_subdirectory("${cppdlr_source}" "${CMAKE_CURRENT_BINARY_DIR}/cppdlr" EXCLUDE_FROM_ALL)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(_core src/_core.cpp)
target_link_libraries(_core PRIVATE cppdlr::cppdlr_c)
install(TARGETS _core LIBRARY DESTINATION pycppdlr COMPONENT python)
