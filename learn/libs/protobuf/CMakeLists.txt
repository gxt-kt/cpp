set(CMAKE_CXX_STANDARD 17)

# include_directories(protobuf)

find_package(Protobuf REQUIRED)  

# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -std=c++11 -lpthread -lprotobuf")  
# generate protobuf cc and h file
# set(PROTO_FILES protobuf.proto)  
# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})  


INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})


# have problem cannot compile for my archlinux
# maybe protobuf version need to change to v3.*
# add_executable(protobuf  protobuf_main.cc)
# target_link_libraries(protobuf ${Protobuf_LIBRARIES})  
# target_include_directories(protobuf PUBLIC ${Protobuf_INCLUDE_DIRS})  
