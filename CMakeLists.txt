cmake_minimum_required(VERSION 3.27)
project(fastcatan CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # GCC/Clang expose a complex-arctangent builtin named `catan`; disable it
    # so the project namespace does not trigger a declaration warning.
    add_compile_options(-fno-builtin-catan)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)

# Tell CMake to enable LTO when building Release.
include(CheckIPOSupported)
check_ipo_supported(RESULT _IPO_SUPPORTED OUTPUT _IPO_LOG)
if(_IPO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# Target CPU. "native" = tune for the build machine (fast local/dev builds).
# For DISTRIBUTED wheels this MUST be a portable baseline, else the wheel emits
# instructions that SIGILL-crash on other CPUs. The PyPI CI build overrides this
# to "x86-64-v2" (SSE4.2 + POPCNT, portable to ~any 2009+ CPU). "off" = no -march.
set(FASTCATAN_ARCH "native" CACHE STRING
    "Value for -march (native | x86-64-v2 | x86-64-v3 | off)")
if(FASTCATAN_ARCH STREQUAL "off")
    set(_FCATAN_ARCH_FLAG "")
else()
    set(_FCATAN_ARCH_FLAG "-march=${FASTCATAN_ARCH}")
endif()
message(STATUS "fastcatan -march=${FASTCATAN_ARCH}")

# Optimized core flags. The Python binding keeps exceptions and RTTI enabled.
set(FCATAN_RELEASE_FLAGS
    -O3 ${_FCATAN_ARCH_FLAG} -fno-exceptions -fno-rtti -Wall -Wextra)

# Core rules library (static; the linker can fold into anything).
add_library(fastcatan_core STATIC
    src/catan/rules.cpp
    src/catan/batched_env.cpp
)
target_include_directories(fastcatan_core PUBLIC include)
target_compile_options(fastcatan_core PRIVATE
    $<$<CONFIG:Release>:${FCATAN_RELEASE_FLAGS}>)
set_target_properties(fastcatan_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

# OpenMP — optional. Macs need libomp from homebrew; Linux GCC has it built-in.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fastcatan_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(fastcatan_core PUBLIC FCATAN_HAVE_OPENMP=1)
    message(STATUS "OpenMP enabled")
else()
    message(STATUS "OpenMP not found — building single-threaded")
endif()

if(NOT SKBUILD)
    # Standalone invariant harness. It plays legal games in pure C++ and
    # checks resource, piece, phase, and terminal-state invariants.
    add_executable(fuzz_invariants tests/fuzz_invariants.cpp)
    target_link_libraries(fuzz_invariants PRIVATE fastcatan_core)
    target_compile_options(fuzz_invariants PRIVATE
        $<$<CONFIG:Release>:${FCATAN_RELEASE_FLAGS}>)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(fuzz_invariants PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_executable(test_outcomes tests/test_outcomes.cpp)
    target_link_libraries(test_outcomes PRIVATE fastcatan_core)
    target_compile_options(test_outcomes PRIVATE
        $<$<CONFIG:Release>:${FCATAN_RELEASE_FLAGS}>)

    enable_testing()
    add_test(NAME invariants COMMAND fuzz_invariants 100000)
    add_test(NAME outcomes COMMAND test_outcomes)
    set_tests_properties(invariants PROPERTIES
        TIMEOUT 900
        PASS_REGULAR_EXPRESSION "invariant failures: 0")
endif()

# --------------------------------------------------------------------
# Python extension via nanobind (built when invoked through scikit-build-core).
# --------------------------------------------------------------------
if(SKBUILD)
    # Development.SABIModule is REQUIRED for nanobind STABLE_ABI to actually emit a
    # limited-API `_fastcatan.abi3.so`. Without it nanobind silently falls back to a
    # version-specific `.cpython-3XX.so`, which imports only on the build minor.
    find_package(Python 3.12 REQUIRED
        COMPONENTS Interpreter Development.Module Development.SABIModule)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_fastcatan
        STABLE_ABI
        NB_STATIC
        bindings/pycatan/bindings.cpp
    )
    target_link_libraries(_fastcatan PRIVATE fastcatan_core)
    # Nanobind needs RTTI + exceptions, so DON'T apply our -fno-* flags here.
    # Core library is still built without them; this binding TU compiles
    # with the compiler defaults.
    target_compile_options(_fastcatan PRIVATE -O3 ${_FCATAN_ARCH_FLAG})

    install(TARGETS _fastcatan LIBRARY DESTINATION fastcatan)
    install(FILES python/fastcatan/__init__.py
            DESTINATION fastcatan)
endif()

# --------------------------------------------------------------------
# IDE indexing target for bindings.cpp (clangd / VS Code).
# Not built by default; gives compile_commands.json an entry so clangd
# can resolve nanobind + Python headers. Skipped when configured via
# scikit-build (the real Python module is built there).
# --------------------------------------------------------------------
if(NOT SKBUILD)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module QUIET)
    if(Python_FOUND)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import nanobind, sys; sys.stdout.write(nanobind.include_dir())"
            OUTPUT_VARIABLE _NB_INCLUDE
            RESULT_VARIABLE _NB_RC
        )
        if(_NB_RC EQUAL 0 AND EXISTS "${_NB_INCLUDE}")
            add_library(bindings_ide OBJECT EXCLUDE_FROM_ALL
                bindings/pycatan/bindings.cpp)
            target_include_directories(bindings_ide PRIVATE
                ${_NB_INCLUDE}
                ${Python_INCLUDE_DIRS})
            target_link_libraries(bindings_ide PRIVATE fastcatan_core)
            target_compile_options(bindings_ide PRIVATE -O0)
            message(STATUS "IDE indexing target enabled for bindings.cpp")
        else()
            message(STATUS "nanobind not found in active Python; bindings.cpp IDE target skipped")
        endif()
    endif()
endif()
