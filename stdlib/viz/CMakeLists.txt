set(SOURCE_FILES
  draw_box_kernel_cpu.cpp)

if (BUILD_CUDA)
  list(APPEND SOURCE_FILES)
endif()

add_library(viz OBJECT ${SOURCE_FILES})

list(APPEND OPENCV_COMPONENTS core highgui imgproc)
set(OPENCV_COMPONENTS ${OPENCV_COMPONENTS} PARENT_SCOPE)

set(STDLIB_LIBRARIES ${STDLIB_LIBRARIES} PARENT_SCOPE)
