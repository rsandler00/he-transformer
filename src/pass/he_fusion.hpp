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

#include "ngraph/pass/graph_rewrite.hpp"

namespace ngraph {
namespace he {
namespace pass {

class HEFusion : public ngraph::pass::GraphRewrite {
 public:
  HEFusion() : GraphRewrite() {
    construct_bounded_relu();
    insert_rescale_after_dot();
    insert_rescale_after_multiply();
    insert_rescale_after_conv();
    insert_rescale_after_avg_pool();
    merge_rescales();
  }

  void construct_bounded_relu();
  void insert_rescale_after_dot();
  void insert_rescale_after_multiply();
  void insert_rescale_after_conv();
  void insert_rescale_after_avg_pool();
  void merge_rescales();
};
}  // namespace pass
}  // namespace he
}  // namespace ngraph
