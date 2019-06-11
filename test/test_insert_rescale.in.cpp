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

#include "ngraph/ngraph.hpp"
#include "op/rescale.hpp"
#include "pass/insert_rescale.hpp"
#include "seal/he_seal_backend.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, insert_rescale_after_multiply_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape{2, 3};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Multiply>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  // Create some tensors for input/output
  auto t_a = he_backend->create_plain_tensor(element::f32, shape);
  auto t_b = he_backend->create_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_plain_tensor(element::f32, shape);

  copy_data(t_a, vector<float>{-2, -1, 0, 1, 2, 3});
  copy_data(t_b, vector<float>{3, -2, 5, 3, 2, -5});
  auto handle = backend->compile(f);
  EXPECT_EQ(1, count_ops_of_type<op::Rescale>(f));

  handle->call_with_validate({t_result}, {t_a, t_b});
  EXPECT_TRUE(all_close((vector<float>{-6, 2, 0, 3, 4, -15}),
                        read_vector<float>(t_result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, insert_rescale_after_multiply_cipher_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape{2, 3};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Multiply>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  // Create some tensors for input/output
  auto t_a = he_backend->create_cipher_tensor(element::f32, shape);
  auto t_b = he_backend->create_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);

  copy_data(t_a, vector<float>{-2, -1, 0, 1, 2, 3});
  copy_data(t_b, vector<float>{3, -2, 5, 3, 2, -5});
  auto handle = backend->compile(f);
  EXPECT_EQ(1, count_ops_of_type<op::Rescale>(f));

  handle->call_with_validate({t_result}, {t_a, t_b});
  EXPECT_TRUE(all_close((vector<float>{-6, 2, 0, 3, 4, -15}),
                        read_vector<float>(t_result), 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, insert_rescale_after_multiply_cipher_cipher) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape{2, 3};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Multiply>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  // Create some tensors for input/output
  auto t_a = he_backend->create_cipher_tensor(element::f32, shape);
  auto t_b = he_backend->create_cipher_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);

  copy_data(t_a, vector<float>{-2, -1, 0, 1, 2, 3});
  copy_data(t_b, vector<float>{3, -2, 5, 3, 2, -5});
  auto handle = backend->compile(f);
  EXPECT_EQ(1, count_ops_of_type<op::Rescale>(f));

  handle->call_with_validate({t_result}, {t_a, t_b});
  EXPECT_TRUE(all_close((vector<float>{-6, 2, 0, 3, 4, -15}),
                        read_vector<float>(t_result), 1e-3f));
}
