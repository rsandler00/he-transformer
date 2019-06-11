//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/pad.hpp"
#include "seal/he_seal_backend.hpp"

namespace ngraph {
namespace he {
void rescale_seal(const std::vector<HEPlaintext>& arg,
                  std::vector<HEPlaintext>& out) {
  NGRAPH_CHECK(arg.size() == out.size(), "arg.size() != out.size() in rescale");
#pragma omp parallel for
  for (size_t i = 0; i < arg.size(); ++i) {
    out[i] = arg[i];
  }
}
}  // namespace he
}  // namespace ngraph