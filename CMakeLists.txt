cmake_minimum_required(VERSION 3.6)
project(background_mask)
find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
    main.cpp
    Makefile)

add_executable(background_mask ${SOURCE_FILES})
TARGET_LINK_LIBRARIES(background_mask -lm -ldl ${OpenCV_LIBS})