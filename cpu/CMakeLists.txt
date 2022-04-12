project(CPU_DE)

set(CMAKE_CXX_STANDARD 17)

file(GLOB SOURCE "./*.cpp" "./*.h" "DE/*.h")

add_library(CPU_DE SHARED ${SOURCE})

target_include_directories(CPU_DE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})