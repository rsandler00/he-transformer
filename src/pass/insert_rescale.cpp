//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/op/add.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/util.hpp"
#include "pass/insert_rescale.hpp"

using namespace std;
using namespace ngraph;

#define TI(x) std::type_index(typeid(x))

static bool rescale_after_avg_pool(const std::shared_ptr<Node>& node) {
  NGRAPH_INFO << "Rescale after avgpool";
  return true;
}

static bool rescale_after_conv(const std::shared_ptr<Node>& node) {
  NGRAPH_INFO << "Rescale after conv";
  return true;
  /* auto broadcast = std::static_pointer_cast<op::Broadcast>(node);
  if (broadcast->get_input_shape(0) == broadcast->get_output_shape(0)) {
    replace_node(node, node->get_argument(0));
    return true;
  }
  return false; */
}

static bool rescale_after_dot(const std::shared_ptr<Node>& node) {
  NGRAPH_INFO << "Rescale after dot";
  return true;
}
static bool rescale_after_multiply(const std::shared_ptr<Node>& node) {
  NGRAPH_INFO << "Rescale after mult";
  return true;
}

static const std::unordered_map<
    std::type_index, std::function<bool(const std::shared_ptr<Node>&)>>
    dispatcher{{TI(op::AvgPool), &rescale_after_avg_pool},
               {TI(op::Convolution), &rescale_after_conv},
               {TI(op::Dot), &rescale_after_dot},
               {TI(op::Multiply), &rescale_after_multiply}};

bool ngraph::he::pass::InsertRescale::run_on_function(
    std::shared_ptr<Function> function) {
  bool clobbered = false;

  for (const auto& n : function->get_ops()) {
    // Work around a warning [-Wpotentially-evaluated-expression]
    const Node& node = *n;
    auto handler = dispatcher.find(TI(node));
    if (handler != dispatcher.end()) {
      clobbered = handler->second(n) || clobbered;
    }
  }
  return clobbered;
}