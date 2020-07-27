# Original: https://gitlab.kitware.com/cmake/cmake/raw/v3.9.6/Modules/FindProtobuf.cmake
# Only minor modifications for DALI
#
# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#.rst:
# FindProtobuf
# ------------
#
# Locate and configure the Google Protocol Buffers library.
#
# The following variables can be set and are optional:
#
# ``Protobuf_SRC_ROOT_FOLDER``
#   When compiling with MSVC, if this cache variable is set
#   the protobuf-default VS project build locations
#   (vsprojects/Debug and vsprojects/Release
#   or vsprojects/x64/Debug and vsprojects/x64/Release)
#   will be searched for libraries and binaries.
# ``Protobuf_IMPORT_DIRS``
#   List of additional directories to be searched for
#   imported .proto files.
# ``Protobuf_DEBUG``
#   Show debug messages.
# ``Protobuf_USE_STATIC_LIBS``
#   Set to ON to force the use of the static libraries.
#   Default is OFF.
# ``Protobuf_CROSS``
#   Search for Protobuf libraries and includes with NO_SYSTEM_ENVIRONMENT_PATH
#   argument - effectively skipping the standard system environment variables
#   and allowing to find protobuf in CMAKE_SYSTEM_PREFIX_PATH set by toolchain
#   definition. protoc binary is taken from native system.
#
# Defines the following variables:
#
# ``Protobuf_FOUND``
#   Found the Google Protocol Buffers library
#   (libprotobuf & header files)
# ``Protobuf_VERSION``
#   Version of package found.
# ``Protobuf_INCLUDE_DIRS``
#   Include directories for Google Protocol Buffers
# ``Protobuf_LIBRARIES``
#   The protobuf libraries
# ``Protobuf_PROTOC_LIBRARIES``
#   The protoc libraries
# ``Protobuf_LITE_LIBRARIES``
#   The protobuf-lite libraries
#
# The following :prop_tgt:`IMPORTED` targets are also defined:
#
# ``protobuf::libprotobuf``
#   The protobuf library.
# ``protobuf::libprotobuf-lite``
#   The protobuf lite library.
# ``protobuf::libprotoc``
#   The protoc library.
#
# The following cache variables are also available to set or use:
#
# ``Protobuf_LIBRARY``
#   The protobuf library
# ``Protobuf_PROTOC_LIBRARY``
#   The protoc library
# ``Protobuf_INCLUDE_DIR``
#   The include directory for protocol buffers
# ``Protobuf_PROTOC_EXECUTABLE``
#   The protoc compiler
# ``Protobuf_LIBRARY_DEBUG``
#   The protobuf library (debug)
# ``Protobuf_PROTOC_LIBRARY_DEBUG``
#   The protoc library (debug)
# ``Protobuf_LITE_LIBRARY``
#   The protobuf lite library
# ``Protobuf_LITE_LIBRARY_DEBUG``
#   The protobuf lite library (debug)
#
# Example:
#
# .. code-block:: cmake
#
#   find_package(Protobuf REQUIRED)
#   include_directories(${Protobuf_INCLUDE_DIRS})
#   include_directories(${CMAKE_CURRENT_BINARY_DIR})
#   protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS foo.proto)
#   protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS EXPORT_MACRO DLL_EXPORT foo.proto)
#   protobuf_generate_python(PROTO_PY foo.proto)
#   add_executable(bar bar.cc ${PROTO_SRCS} ${PROTO_HDRS})
#   target_link_libraries(bar ${Protobuf_LIBRARIES})
#
# .. note::
#   The ``protobuf_generate_cpp`` and ``protobuf_generate_python``
#   functions and :command:`add_executable` or :command:`add_library`
#   calls only work properly within the same directory.
#
# .. command:: protobuf_generate_cpp
#
#   Add custom commands to process ``.proto`` files to C++::
#
#     protobuf_generate_cpp (<SRCS> <HDRS> [EXPORT_MACRO <MACRO>] [<ARGN>...])
#
#   ``SRCS``
#     Variable to define with autogenerated source files
#   ``HDRS``
#     Variable to define with autogenerated header files
#   ``EXPORT_MACRO``
#     is a macro which should expand to ``__declspec(dllexport)`` or
#     ``__declspec(dllimport)`` depending on what is being compiled.
#   ``ARGN``
#     ``.proto`` files
#
# .. command:: protobuf_generate_python
#
#   Add custom commands to process ``.proto`` files to Python::
#
#     protobuf_generate_python (<PY> [<ARGN>...])
#
#   ``PY``
#     Variable to define with autogenerated Python files
#   ``ARGN``
#     ``.proto`` filess

function(PROTOBUF_GENERATE_CPP SRCS HDRS)
  cmake_parse_arguments(protobuf "" "EXPORT_MACRO" "" ${ARGN})

  set(PROTO_FILES "${protobuf_UNPARSED_ARGUMENTS}")
  if(NOT PROTO_FILES)
    message(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called without any proto files")
    return()
  endif()

  if(protobuf_EXPORT_MACRO)
    set(DLL_EXPORT_DECL "dllexport_decl=${protobuf_EXPORT_MACRO}:")
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(FIL ${PROTO_FILES})
      get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
      get_filename_component(ABS_PATH ${ABS_FIL} PATH)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  else()
    set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS AND NOT DEFINED Protobuf_IMPORT_DIRS)
    set(Protobuf_IMPORT_DIRS "${PROTOBUF_IMPORT_DIRS}")
  endif()

  if(DEFINED Protobuf_IMPORT_DIRS)
    foreach(DIR ${Protobuf_IMPORT_DIRS})
      get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  endif()

  set(${SRCS})
  set(${HDRS})
  foreach(FIL ${PROTO_FILES})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    if(NOT PROTOBUF_GENERATE_CPP_APPEND_PATH)
      get_filename_component(FIL_DIR ${FIL} DIRECTORY)
      if(FIL_DIR)
        set(FIL_WE "${FIL_DIR}/${FIL_WE}")
      endif()
    endif()

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc"
             "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h"
      COMMAND  ${Protobuf_PROTOC_EXECUTABLE}
      ARGS "--cpp_out=${DLL_EXPORT_DECL}${CMAKE_CURRENT_BINARY_DIR}" ${_protobuf_include_path} ${ABS_FIL}
      DEPENDS ${ABS_FIL} ${Protobuf_PROTOC_EXECUTABLE}
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

function(PROTOBUF_GENERATE_PYTHON SRCS)
  if(NOT ARGN)
    message(SEND_ERROR "Error: PROTOBUF_GENERATE_PYTHON() called without any proto files")
    return()
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(FIL ${ARGN})
      get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
      get_filename_component(ABS_PATH ${ABS_FIL} PATH)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  else()
    set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS AND NOT DEFINED Protobuf_IMPORT_DIRS)
    set(Protobuf_IMPORT_DIRS "${PROTOBUF_IMPORT_DIRS}")
  endif()

  if(DEFINED Protobuf_IMPORT_DIRS)
    foreach(DIR ${Protobuf_IMPORT_DIRS})
      get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  endif()

  set(${SRCS})
  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    if(NOT PROTOBUF_GENERATE_CPP_APPEND_PATH)
      get_filename_component(FIL_DIR ${FIL} DIRECTORY)
      if(FIL_DIR)
        set(FIL_WE "${FIL_DIR}/${FIL_WE}")
      endif()
    endif()

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_pb2.py")
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_pb2.py"
      COMMAND  ${Protobuf_PROTOC_EXECUTABLE} --python_out ${CMAKE_CURRENT_BINARY_DIR} ${_protobuf_include_path} ${ABS_FIL}
      DEPENDS ${ABS_FIL} ${Protobuf_PROTOC_EXECUTABLE}
      COMMENT "Running Python protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
endfunction()


if(Protobuf_DEBUG)
  # Output some of their choices
  message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                 "Protobuf_USE_STATIC_LIBS = ${Protobuf_USE_STATIC_LIBS}")
endif()


# Backwards compatibility
# Define camel case versions of input variables
foreach(UPPER
    PROTOBUF_SRC_ROOT_FOLDER
    PROTOBUF_IMPORT_DIRS
    PROTOBUF_DEBUG
    PROTOBUF_LIBRARY
    PROTOBUF_PROTOC_LIBRARY
    PROTOBUF_INCLUDE_DIR
    PROTOBUF_PROTOC_EXECUTABLE
    PROTOBUF_LIBRARY_DEBUG
    PROTOBUF_PROTOC_LIBRARY_DEBUG
    PROTOBUF_LITE_LIBRARY
    PROTOBUF_LITE_LIBRARY_DEBUG
    )
    if (DEFINED ${UPPER})
        string(REPLACE "PROTOBUF_" "Protobuf_" Camel ${UPPER})
        if (NOT DEFINED ${Camel})
            set(${Camel} ${${UPPER}})
        endif()
    endif()
endforeach()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(_PROTOBUF_ARCH_DIR x64/)
endif()


# Support preference of static libs by adjusting CMAKE_FIND_LIBRARY_SUFFIXES
if( Protobuf_USE_STATIC_LIBS )
  set( _protobuf_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  if(WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a )
  endif()
endif()

#include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
include(SelectLibraryConfigurations)

if(Protobuf_CROSS)
  set(FIND_PATH_SELECTOR NO_SYSTEM_ENVIRONMENT_PATH)
endif()

# Internal function: search for normal library as well as a debug one
#    if the debug one is specified also include debug/optimized keywords
#    in *_LIBRARIES variable
function(_protobuf_find_libraries name filename)
  if(${name}_LIBRARIES)
    # Use result recorded by a previous call.
    return()
  elseif(${name}_LIBRARY)
    # Honor cache entry used by CMake 3.5 and lower.
    set(${name}_LIBRARIES "${${name}_LIBRARY}" PARENT_SCOPE)
  else()
    find_library(${name}_LIBRARY_RELEASE
      NAMES ${filename}
      PATHS ${Protobuf_SRC_ROOT_FOLDER}/vsprojects/${_PROTOBUF_ARCH_DIR}Release ${FIND_PATH_SELECTOR})
    mark_as_advanced(${name}_LIBRARY_RELEASE)

    find_library(${name}_LIBRARY_DEBUG
      NAMES ${filename}d ${filename}
      PATHS ${Protobuf_SRC_ROOT_FOLDER}/vsprojects/${_PROTOBUF_ARCH_DIR}Debug ${FIND_PATH_SELECTOR})
    mark_as_advanced(${name}_LIBRARY_DEBUG)

    select_library_configurations(${name})
    set(${name}_LIBRARY "${${name}_LIBRARY}" PARENT_SCOPE)
    set(${name}_LIBRARIES "${${name}_LIBRARIES}" PARENT_SCOPE)
  endif()
endfunction()

# Internal function: find threads library
function(_protobuf_find_threads)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    find_package(Threads)
    if(Threads_FOUND)
        list(APPEND Protobuf_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
        set(Protobuf_LIBRARIES "${Protobuf_LIBRARIES}" PARENT_SCOPE)
    endif()
endfunction()

#
# Main.
#

# By default have PROTOBUF_GENERATE_CPP macro pass -I to protoc
# for each directory where a proto file is referenced.
if(NOT DEFINED PROTOBUF_GENERATE_CPP_APPEND_PATH)
  set(PROTOBUF_GENERATE_CPP_APPEND_PATH TRUE)
endif()


# Google's provided vcproj files generate libraries with a "lib"
# prefix on Windows
if(MSVC)
    set(Protobuf_ORIG_FIND_LIBRARY_PREFIXES "${CMAKE_FIND_LIBRARY_PREFIXES}")
    set(CMAKE_FIND_LIBRARY_PREFIXES "lib" "")

    find_path(Protobuf_SRC_ROOT_FOLDER protobuf.pc.in)
endif()

# The Protobuf library
_protobuf_find_libraries(Protobuf protobuf)
#DOC "The Google Protocol Buffers RELEASE Library"

_protobuf_find_libraries(Protobuf_LITE protobuf-lite)

# The Protobuf Protoc Library
_protobuf_find_libraries(Protobuf_PROTOC protoc)

if(Protobuf_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
        "Found libraries path are: ${Protobuf_LIBRARY}, ${Protobuf_LITE_LIBRARY}, ${Protobuf_PROTOC_LIBRARY}")
endif()

# Restore original find library prefixes
if(MSVC)
    set(CMAKE_FIND_LIBRARY_PREFIXES "${Protobuf_ORIG_FIND_LIBRARY_PREFIXES}")
endif()

if(UNIX)
    _protobuf_find_threads()
endif()

# Find the include directory
find_path(Protobuf_INCLUDE_DIR
    google/protobuf/service.h
    PATHS ${Protobuf_SRC_ROOT_FOLDER}/src
    ${FIND_PATH_SELECTOR}
)
mark_as_advanced(Protobuf_INCLUDE_DIR)

# Find the protoc Executable
# We always look for the native version
find_program(Protobuf_PROTOC_EXECUTABLE
    NAMES protoc
    DOC "The Google Protocol Buffers Compiler"
    PATHS
    ${Protobuf_SRC_ROOT_FOLDER}/vsprojects/${_PROTOBUF_ARCH_DIR}Release
    ${Protobuf_SRC_ROOT_FOLDER}/vsprojects/${_PROTOBUF_ARCH_DIR}Debug
    NO_CMAKE_SYSTEM_PATH
)
mark_as_advanced(Protobuf_PROTOC_EXECUTABLE)

if(Protobuf_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
        "requested version of Google Protobuf is ${Protobuf_FIND_VERSION}")
endif()

if(Protobuf_INCLUDE_DIR)
  set(_PROTOBUF_COMMON_HEADER ${Protobuf_INCLUDE_DIR}/google/protobuf/stubs/common.h)

  if(Protobuf_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "location of common.h: ${_PROTOBUF_COMMON_HEADER}")
  endif()

  set(Protobuf_VERSION "")
  set(Protobuf_LIB_VERSION "")
  file(STRINGS ${_PROTOBUF_COMMON_HEADER} _PROTOBUF_COMMON_H_CONTENTS REGEX "#define[ \t]+GOOGLE_PROTOBUF_VERSION[ \t]+")
  if(_PROTOBUF_COMMON_H_CONTENTS MATCHES "#define[ \t]+GOOGLE_PROTOBUF_VERSION[ \t]+([0-9]+)")
      set(Protobuf_LIB_VERSION "${CMAKE_MATCH_1}")
  endif()
  unset(_PROTOBUF_COMMON_H_CONTENTS)

  math(EXPR _PROTOBUF_MAJOR_VERSION "${Protobuf_LIB_VERSION} / 1000000")
  math(EXPR _PROTOBUF_MINOR_VERSION "${Protobuf_LIB_VERSION} / 1000 % 1000")
  math(EXPR _PROTOBUF_SUBMINOR_VERSION "${Protobuf_LIB_VERSION} % 1000")
  set(Protobuf_VERSION "${_PROTOBUF_MAJOR_VERSION}.${_PROTOBUF_MINOR_VERSION}.${_PROTOBUF_SUBMINOR_VERSION}")

  if(Protobuf_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
        "${_PROTOBUF_COMMON_HEADER} reveals protobuf ${Protobuf_VERSION}")
  endif()

  # Check Protobuf compiler version to be aligned with libraries version
  execute_process(COMMAND ${Protobuf_PROTOC_EXECUTABLE} --version
                  OUTPUT_VARIABLE _PROTOBUF_PROTOC_EXECUTABLE_VERSION)

  if("${_PROTOBUF_PROTOC_EXECUTABLE_VERSION}" MATCHES "libprotoc ([0-9.]+)")
    set(_PROTOBUF_PROTOC_EXECUTABLE_VERSION "${CMAKE_MATCH_1}")
  endif()

  if(Protobuf_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
        "${Protobuf_PROTOC_EXECUTABLE} reveals version ${_PROTOBUF_PROTOC_EXECUTABLE_VERSION}")
  endif()

  if(NOT "${_PROTOBUF_PROTOC_EXECUTABLE_VERSION}" VERSION_EQUAL "${Protobuf_VERSION}")
      message(WARNING "Protobuf compiler version ${_PROTOBUF_PROTOC_EXECUTABLE_VERSION}"
          " doesn't match library version ${Protobuf_VERSION}")
  endif()

  if(Protobuf_LIBRARY)
      if(NOT TARGET protobuf::libprotobuf)
          add_library(protobuf::libprotobuf UNKNOWN IMPORTED)
          set_target_properties(protobuf::libprotobuf PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIR}")
          if(EXISTS "${Protobuf_LIBRARY}")
            set_target_properties(protobuf::libprotobuf PROPERTIES
              IMPORTED_LOCATION "${Protobuf_LIBRARY}")
          endif()
          if(EXISTS "${Protobuf_LIBRARY_RELEASE}")
            set_property(TARGET protobuf::libprotobuf APPEND PROPERTY
              IMPORTED_CONFIGURATIONS RELEASE)
            set_target_properties(protobuf::libprotobuf PROPERTIES
              IMPORTED_LOCATION_RELEASE "${Protobuf_LIBRARY_RELEASE}")
          endif()
          if(EXISTS "${Protobuf_LIBRARY_DEBUG}")
            set_property(TARGET protobuf::libprotobuf APPEND PROPERTY
              IMPORTED_CONFIGURATIONS DEBUG)
            set_target_properties(protobuf::libprotobuf PROPERTIES
              IMPORTED_LOCATION_DEBUG "${Protobuf_LIBRARY_DEBUG}")
          endif()
      endif()
  endif()

  if(Protobuf_LITE_LIBRARY)
      if(NOT TARGET protobuf::libprotobuf-lite)
          add_library(protobuf::libprotobuf-lite UNKNOWN IMPORTED)
          set_target_properties(protobuf::libprotobuf-lite PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIR}")
          if(EXISTS "${Protobuf_LITE_LIBRARY}")
            set_target_properties(protobuf::libprotobuf-lite PROPERTIES
              IMPORTED_LOCATION "${Protobuf_LITE_LIBRARY}")
          endif()
          if(EXISTS "${Protobuf_LITE_LIBRARY_RELEASE}")
            set_property(TARGET protobuf::libprotobuf-lite APPEND PROPERTY
              IMPORTED_CONFIGURATIONS RELEASE)
            set_target_properties(protobuf::libprotobuf-lite PROPERTIES
              IMPORTED_LOCATION_RELEASE "${Protobuf_LITE_LIBRARY_RELEASE}")
          endif()
          if(EXISTS "${Protobuf_LITE_LIBRARY_DEBUG}")
            set_property(TARGET protobuf::libprotobuf-lite APPEND PROPERTY
              IMPORTED_CONFIGURATIONS DEBUG)
            set_target_properties(protobuf::libprotobuf-lite PROPERTIES
              IMPORTED_LOCATION_DEBUG "${Protobuf_LITE_LIBRARY_DEBUG}")
          endif()
      endif()
  endif()

  if(Protobuf_PROTOC_LIBRARY)
      if(NOT TARGET protobuf::libprotoc)
          add_library(protobuf::libprotoc UNKNOWN IMPORTED)
          set_target_properties(protobuf::libprotoc PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIR}")
          if(EXISTS "${Protobuf_PROTOC_LIBRARY}")
            set_target_properties(protobuf::libprotoc PROPERTIES
              IMPORTED_LOCATION "${Protobuf_PROTOC_LIBRARY}")
          endif()
          if(EXISTS "${Protobuf_PROTOC_LIBRARY_RELEASE}")
            set_property(TARGET protobuf::libprotoc APPEND PROPERTY
              IMPORTED_CONFIGURATIONS RELEASE)
            set_target_properties(protobuf::libprotoc PROPERTIES
              IMPORTED_LOCATION_RELEASE "${Protobuf_PROTOC_LIBRARY_RELEASE}")
          endif()
          if(EXISTS "${Protobuf_PROTOC_LIBRARY_DEBUG}")
            set_property(TARGET protobuf::libprotoc APPEND PROPERTY
              IMPORTED_CONFIGURATIONS DEBUG)
            set_target_properties(protobuf::libprotoc PROPERTIES
              IMPORTED_LOCATION_DEBUG "${Protobuf_PROTOC_LIBRARY_DEBUG}")
          endif()
      endif()
  endif()
endif()

#include(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Protobuf
    REQUIRED_VARS Protobuf_LIBRARIES Protobuf_INCLUDE_DIR
    VERSION_VAR Protobuf_VERSION
)

if(Protobuf_FOUND)
    set(Protobuf_INCLUDE_DIRS ${Protobuf_INCLUDE_DIR})
endif()

# Restore the original find library ordering
if( Protobuf_USE_STATIC_LIBS )
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_protobuf_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

# Backwards compatibility
# Define upper case versions of output variables
foreach(Camel
    Protobuf_SRC_ROOT_FOLDER
    Protobuf_IMPORT_DIRS
    Protobuf_DEBUG
    Protobuf_INCLUDE_DIRS
    Protobuf_LIBRARIES
    Protobuf_PROTOC_LIBRARIES
    Protobuf_LITE_LIBRARIES
    Protobuf_LIBRARY
    Protobuf_PROTOC_LIBRARY
    Protobuf_INCLUDE_DIR
    Protobuf_PROTOC_EXECUTABLE
    Protobuf_LIBRARY_DEBUG
    Protobuf_PROTOC_LIBRARY_DEBUG
    Protobuf_LITE_LIBRARY
    Protobuf_LITE_LIBRARY_DEBUG
    )
    string(TOUPPER ${Camel} UPPER)
    set(${UPPER} ${${Camel}})
endforeach()
