project(imageBasics)

add_executable(imageBasics imageBasics.cpp)
# 链接OpenCV库
target_link_libraries(imageBasics ${OpenCV_LIBS})

add_executable(imageBasicstest imageBasicstest.cpp)
# 链接OpenCV库
target_link_libraries(imageBasicstest ${OpenCV_LIBS})

add_executable(undistortImage undistortImage.cpp)
# 链接OpenCV库
target_link_libraries(undistortImage ${OpenCV_LIBS})

add_executable(undistortImagetest undistortImagetest.cpp)
# 链接OpenCV库
target_link_libraries(undistortImagetest ${OpenCV_LIBS})
