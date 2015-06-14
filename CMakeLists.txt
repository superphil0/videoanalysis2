cmake_minimum_required(VERSION 3.1)
project(VideoAnalysis)

set(OpenCV_DIR C:\\Code\\opencv\\build)
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})


SET(CMAKE_CXX_FLAGS "-std=c++11")

# ===
# executable
# ===

include_directories(
	${CMAKE_SOURCE_DIR}/src
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

file(GLOB_RECURSE SRC_FILES
	${CMAKE_SOURCE_DIR}/src/*.cpp
	${CMAKE_SOURCE_DIR}/src/*.h
) 

add_executable(VideoAnalysis ${SRC_FILES})

target_link_libraries(VideoAnalysis ${OpenCV_LIBS})

# ===
# source groups
# ===

function(create_source_groups DIR PARENT_GROUP)
    file (GLOB SUB_DIRS "${DIR}/[^.]*")
    foreach (SUB_DIR ${SUB_DIRS})
        get_filename_component(SUB_DIR_NAME ${SUB_DIR} NAME)
        if(IS_DIRECTORY ${SUB_DIR})
            if (PARENT_GROUP STREQUAL "")
                source_group("${SUB_DIR_NAME}" ${SUB_DIR}/*)
                create_source_groups("${SUB_DIR}" "${SUB_DIR_NAME}")
            else()
                source_group("${PARENT_GROUP}\\${SUB_DIR_NAME}" ${SUB_DIR}/*)
                create_source_groups("${SUB_DIR}" "${PARENT_GROUP}\\${SUB_DIR_NAME}")
            endif()
        endif()
    endforeach()
endfunction()

create_source_groups("${CMAKE_SOURCE_DIR}/src" "src")
create_source_groups("${CMAKE_SOURCE_DIR}/libs" "libs")

source_group("src" src/*)