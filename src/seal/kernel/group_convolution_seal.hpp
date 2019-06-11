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

#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/negate_seal.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace he {
void group_convolution_seal(
    const std::vector<HEPlaintext>& arg0, const std::vector<HEPlaintext>& arg1,
    std::vector<HEPlaintext>& out, const Shape& arg0_shape,
    const Shape& arg1_shape, const Shape& out_shape,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides, const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above, const Strides& data_dilation_strides,
    const element::Type& element_type,
    const ngraph::he::HESealBackend& he_seal_backend) {
  NGRAPH_INFO << "arg0_shape " << join(arg0_shape, "x");
  NGRAPH_INFO << "arg1_shape " << join(arg1_shape, "x");
  NGRAPH_INFO << "out_shape " << join(out_shape, "x");
  NGRAPH_INFO << "window_movement_strides "
              << join(window_movement_strides, "x");
  NGRAPH_INFO << "window_dilation_strides "
              << join(window_dilation_strides, "x");
  NGRAPH_INFO << "padding_below " << padding_below;
  NGRAPH_INFO << "padding_above " << padding_above;
  NGRAPH_INFO << "data_dilation_strides " << data_dilation_strides;

  throw ngraph_error("Group convolution unimplemented");
}
}  // namespace he
}  // namespace ngraph
