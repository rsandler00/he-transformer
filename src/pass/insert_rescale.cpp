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

#include "ngraph/check.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/util.hpp"
#include "op/rescale.hpp"
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
  auto old_node = std::static_pointer_cast<op::Convolution>(node);
  shared_ptr<Node> new_node = old_node->copy_with_new_args(
      NodeVector{old_node->get_argument(0), old_node->get_argument(1)});
  auto rescale_node = make_shared<op::Rescale>(new_node);
  replace_node(old_node, rescale_node);
  return true;
}

static bool rescale_after_dot(const std::shared_ptr<Node>& node) {
  NGRAPH_INFO << "Rescale after dot";
  auto old_node = std::static_pointer_cast<op::Dot>(node);
  shared_ptr<Node> new_node = old_node->copy_with_new_args(
      NodeVector{old_node->get_argument(0), old_node->get_argument(1)});
  auto rescale_node = make_shared<op::Rescale>(new_node);
  replace_node(old_node, rescale_node);
  return true;
}
static bool rescale_after_multiply(const std::shared_ptr<Node>& node) {
  NGRAPH_INFO << "Rescale after mult";
  auto old_node = std::static_pointer_cast<op::Multiply>(node);
  shared_ptr<Node> new_node = old_node->copy_with_new_args(
      NodeVector{old_node->get_argument(0), old_node->get_argument(1)});
  auto rescale_node = make_shared<op::Rescale>(new_node);
  replace_node(old_node, rescale_node);
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

    NGRAPH_INFO << node.description();

    auto outputs = node.outputs();
    NGRAPH_CHECK(outputs.size() == 1,
                 "node has more than one output in insert_rescale");
    NGRAPH_INFO << "output node " << outputs[0].get_node()->description();

    auto rescale_op =
        dynamic_cast<const ngraph::op::Rescale*>(outputs[0].get_node());
    if (rescale_op != nullptr) {
      NGRAPH_INFO << "Rescale op already there";
    } else {
      NGRAPH_INFO << "Rescale not yet there";
      auto handler = dispatcher.find(TI(node));
      if (handler != dispatcher.end()) {
        clobbered = handler->second(n) || clobbered;
      }
    }
  }
  return clobbered;
}