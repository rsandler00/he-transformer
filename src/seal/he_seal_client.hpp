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

#include <boost/asio.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "client_util.hpp"
#include "seal/seal.h"
#include "tcp/tcp_client.hpp"
#include "tcp/tcp_message.hpp"

namespace ngraph {
namespace he {
class HESealClient {
 public:
  HESealClient(const std::string& hostname, const size_t port,
               const size_t batch_size, const std::vector<float>& inputs);

  ~HESealClient() = default;

  void set_seal_context();

  void handle_message(const ngraph::he::TCPMessage& message);

  void handle_relu_request(const ngraph::he::TCPMessage& message);

  inline void write_message(ngraph::he::TCPMessage&& message) {
    m_tcp_client->write_message(std::move(message));
  }

  inline bool is_done() { return m_is_done; }

  inline std::vector<float> get_results() { return m_results; }

  void close_connection();

  bool complex_packing() const { return m_complex_packing; }

  void decode_to_real_vec(const seal::Plaintext& plain,
                          std::vector<double>& output, bool complex);

 private:
  std::shared_ptr<TCPClient> m_tcp_client;
  seal::EncryptionParameters m_encryption_params{seal::scheme_type::CKKS};
  std::shared_ptr<seal::PublicKey> m_public_key;
  std::shared_ptr<seal::SecretKey> m_secret_key;
  std::shared_ptr<seal::SEALContext> m_context;
  std::shared_ptr<seal::Encryptor> m_encryptor;
  std::shared_ptr<seal::CKKSEncoder> m_ckks_encoder;
  std::shared_ptr<seal::Decryptor> m_decryptor;
  std::shared_ptr<seal::Evaluator> m_evaluator;
  std::shared_ptr<seal::KeyGenerator> m_keygen;
  std::shared_ptr<seal::RelinKeys> m_relin_keys;
  double m_scale;
  size_t m_batch_size;
  bool m_is_done;
  std::vector<float> m_inputs;   // Function inputs
  std::vector<float> m_results;  // Function outputs

  bool m_complex_packing{std::getenv("NGRAPH_COMPLEX_PACK") != nullptr};
};
}  // namespace he
}  // namespace ngraph
