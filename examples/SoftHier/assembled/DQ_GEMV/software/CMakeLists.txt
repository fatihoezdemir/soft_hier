# Specify the source files for the "src" module
set(SRC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.c
)

# Add DEBUG definition conditionally
if(DEFINED ENV{DEBUG} OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
endif()

# Make sources and include paths visible to the main CMakeLists.txt
set(SOURCES ${SRC_SOURCES} PARENT_SCOPE)
set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include PARENT_SCOPE)


