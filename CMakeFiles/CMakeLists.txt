#include (FindOpenMP) 
find_package(PCL 1.8 REQUIRED)
#find_package(OpenMP REQUIRED)
find_package (Threads)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (correspondence_foveated ../correspondence_foveated.cpp)
target_link_libraries (correspondence_foveated ${PCL_LIBRARIES})
target_link_libraries (correspondence_foveated ${CMAKE_THREAD_LIBS_INIT})

##target_link_libraries (main ${CMAKE_THREAD_LIBS_INIT})

SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wcpp")

#if(OPENMP_FOUND)
#message("OPENMP FOUND")
#set (HAVE_OPENMP TRUE)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}#")
#endif()

if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

#add_definitions(-DGMX_OPENMP=ON)
add_definitions(-DGMX_THREAD_MPI=ON -DWITH_TBB=ON=ON)
#add_definitions(-march=core2)

#add_definitions(-Wall -O3 -DNDEBUG -pipe -ffast-math -funroll-loops -ftree-vectorize -fomit-frame-pointer -pipe -mfpmath=sse -mmmx -msse -mtune=core2 -march=core2 -msse2 -msse3 -mssse3 -msse4)
#add_definitions (-Wall -O3 -Winvalid-pch -pipe -funroll-loops -fno-strict-aliasing -lboost_mpi-gcc-mt-1_35)

