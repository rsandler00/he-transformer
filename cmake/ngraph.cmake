# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

include(ExternalProject)

SET(NGRAPH_CMAKE_GIT_REPOSITORY https://github.com/NervanaSystems/ngraph.git)
SET(NGRAPH_CMAKE_GIT_TAG master)

set(EXTERNAL_NGRAPH_INSTALL_DIR ${EXTERNAL_INSTALL_DIR})
set(NGRAPH_CMAKE_PREFIX ext_ngraph)

ExternalProject_Add(
    ext_ngraph
    GIT_REPOSITORY ${NGRAPH_CMAKE_GIT_REPOSITORY}
    GIT_TAG ${NGRAPH_CMAKE_GIT_TAG}
    PREFIX ${NGRAPH_CMAKE_PREFIX}
    # UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DNGRAPH_INSTALL_PREFIX=${EXTERNAL_NGRAPH_INSTALL_DIR}
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DNGRAPH_CPU_ENABLE=FALSE
               -DNGRAPH_UNIT_TEST_ENABLE=FALSE
               -DNGRAPH_TOOLS_ENABLE=FALSE
    BUILD_BYPRODUCTS ${NGRAPH_CMAKE_PREFIX}
    BUILD_ALWAYS 1
)

ExternalProject_Get_Property(ext_ngraph source_dir)
set(NGRAPH_INCLUDE_DIR ${EXTERNAL_NGRAPH_INSTALL_DIR}/include)
set(NGRAPH_LIB_DIR ${EXTERNAL_NGRAPH_INSTALL_DIR}/lib)
set(NGRAPH_TEST_DIR ${source_dir}/test)