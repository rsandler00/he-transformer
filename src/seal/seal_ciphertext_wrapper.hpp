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

#include <array>
#include <memory>

#include "seal/seal.h"

namespace ngraph {
namespace he {
class SealCiphertextWrapper {
 public:
  SealCiphertextWrapper() : m_complex_packing(false), m_is_zero(false) {}

  SealCiphertextWrapper(const seal::Ciphertext& cipher,
                        bool complex_packing = false, bool is_zero = false)
      : m_ciphertext(cipher),
        m_complex_packing(complex_packing),
        m_is_zero(is_zero) {}

  seal::Ciphertext& ciphertext() { return m_ciphertext; }
  const seal::Ciphertext& ciphertext() const { return m_ciphertext; }

  void save(std::ostream& stream) const { m_ciphertext.save(stream); }

  size_t size() const { return m_ciphertext.size(); }

  bool is_zero() const { return m_is_zero; }
  bool& is_zero() { return m_is_zero; }

  double& scale() { return m_ciphertext.scale(); }
  const double scale() const { return m_ciphertext.scale(); }

  bool complex_packing() const { return m_complex_packing; }
  bool& complex_packing() { return m_complex_packing; }

  void load(void* src) {
    seal::parms_id_type parms_id{};
    seal::SEAL_BYTE is_ntt_form_byte;
    uint64_t size64 = 0;
    uint64_t poly_modulus_degree = 0;
    uint64_t coeff_mod_count = 0;
    double scale = 0;

    static constexpr std::array<size_t, 6> offsets = {
        sizeof(seal::parms_id_type),
        sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE),
        sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
            sizeof(uint64_t),
        sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
            2 * sizeof(uint64_t),
        sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
            3 * sizeof(uint64_t),
        sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
            3 * sizeof(uint64_t) + sizeof(double),
    };

    std::memcpy(&parms_id, src, sizeof(seal::parms_id_type));
    std::memcpy(&is_ntt_form_byte, src + offsets[1], sizeof(seal::SEAL_BYTE));
    std::memcpy(&size64, src + offsets[2], sizeof(uint64_t));
    std::memcpy(&poly_modulus_degree, src + offsets[3], sizeof(uint64_t));
    std::memcpy(&coeff_mod_count, src + offsets[4], sizeof(uint64_t));
    std::memcpy(&scale, src + offsets[5], sizeof(double));

    seal::IntArray<seal::Ciphertext::ct_coeff_type> new_data(
        m_ciphertext.pool());

    m_ciphertext.is_ntt_form() =
        (is_ntt_form_byte == seal::SEAL_BYTE(0)) ? false : true;
    m_ciphertext.scale() = scale;

    // TODO: load/ verify context?
    /* seal::EncryptionParameters parms(seal::scheme_type::CKKS);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus();
    auto context = SEAALContext::Create(parms) */

    m_ciphertext.reserve(size);

    // std::memcpy(destination, (void*)cipher.data()), 8 *
    // cipher.uint64_count());
  }

 private:
  bool m_complex_packing;
  bool m_is_zero;
  seal::Ciphertext m_ciphertext;
};

inline size_t ciphertext_size(const seal::Ciphertext& cipher) {
  // TODO: figure out why the extra 8 bytes
  size_t expected_size = 8;
  expected_size += sizeof(seal::parms_id_type);
  expected_size += sizeof(seal::SEAL_BYTE);
  // size64, poly_modulus_degree, coeff_mod_count
  expected_size += 3 * sizeof(uint64_t);
  // scale
  expected_size += sizeof(double);
  // data
  expected_size += 8 * cipher.uint64_count();
  return expected_size;
}

inline void save(const seal::Ciphertext& cipher, void* destination) {
  static constexpr std::array<size_t, 6> offsets = {
      sizeof(seal::parms_id_type),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) + sizeof(uint64_t),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
          2 * sizeof(uint64_t),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
          3 * sizeof(uint64_t),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
          3 * sizeof(uint64_t) + sizeof(double),
  };

  bool is_ntt_form = cipher.is_ntt_form();
  uint64_t size = cipher.size();
  uint64_t polynomial_modulus_degree = cipher.poly_modulus_degree();
  uint64_t coeff_mod_count = cipher.coeff_mod_count();
  double scale = cipher.scale();

  std::memcpy(destination, (void*)&cipher.parms_id(),
              sizeof(seal::parms_id_type));
  std::memcpy(destination + offsets[0], (void*)&is_ntt_form,
              sizeof(seal::SEAL_BYTE));
  std::memcpy(destination + offsets[1], (void*)&size, sizeof(uint64_t));
  std::memcpy(destination + offsets[2], (void*)&polynomial_modulus_degree,
              sizeof(uint64_t));
  std::memcpy(destination + offsets[3], (void*)&coeff_mod_count,
              sizeof(uint64_t));
  std::memcpy(destination + offsets[4], (void*)&cipher.scale(), sizeof(double));
  std::memcpy(destination + offsets[5], (void*)cipher.data(),
              8 * cipher.uint64_count());
}

inline void load(const seal::Ciphertext& cipher, void* src) {}

}  // namespace he
}  // namespace ngraph
