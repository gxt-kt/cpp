# include_directories(include)

# Get a list of all directories under ./test/
file(
  GLOB directories
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/*)
# Loop through each directory
foreach(dir ${directories})
  # Check if the current entry is a directory
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
      # add_subdirectory
      message(STATUS add_subdirectory (${dir}))
      message(STATUS add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}))
      add_subdirectory(${dir})
    endif()
  endif()
endforeach()

