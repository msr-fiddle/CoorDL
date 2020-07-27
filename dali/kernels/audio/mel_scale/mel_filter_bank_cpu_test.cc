// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include <complex>
#include <cmath>
#include "dali/kernels/scratch.h"
#include "dali/kernels/audio/mel_scale/mel_scale.h"
#include "dali/kernels/audio/mel_scale/mel_filter_bank_cpu.h"
#include "dali/kernels/common/utils.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace kernels {
namespace audio {
namespace test {

class MelScaleCpuTest : public::testing::TestWithParam<
  std::tuple<std::array<int64_t, 2>, /* data_shape */
             int, /* nfilter */
             float, /* sample_rate */
             float, /* fmin */
             float>> {  /* fmax */
 public:
  MelScaleCpuTest()
    : data_shape_(std::get<0>(GetParam()))
    , nfilter_(std::get<1>(GetParam()))
    , sample_rate_(std::get<2>(GetParam()))
    , freq_low_(std::get<3>(GetParam()))
    , freq_high_(std::get<4>(GetParam()))
    , data_(volume(data_shape_))
    , in_view_(data_.data(), data_shape_) {}

  ~MelScaleCpuTest() override = default;

 protected:
  void SetUp() final {
    std::mt19937 rng;
    UniformRandomFill(in_view_, rng, 0.0, 1.0);
  }
  TensorShape<2> data_shape_;
  int nfilter_ = 4;
  float sample_rate_ = 16000, freq_low_ = 0, freq_high_ = 8000;
  std::vector<float> data_;
  OutTensorCPU<float, 2> in_view_;
};

template <typename T>
void print_data(const OutTensorCPU<T, 2>& data_view) {
  auto sh = data_view.shape;
  for (int i0 = 0; i0 < sh[0]; i0++) {
    for (int i1 = 0; i1 < sh[1]; i1++) {
      int k = i0 * sh[1] + i1;
      LOG_LINE << " " << data_view.data[k];
    }
    LOG_LINE << "\n";
  }
}

std::vector<std::vector<float>> ReferenceFilterBanks(int nfilter, int nfft, float sample_rate,
                                                     float low_freq, float high_freq) {
  HtkMelScale<float> mel_scale;

  std::vector<std::vector<float>> fbanks(nfilter);
  auto low_mel = mel_scale.hz_to_mel(low_freq);
  auto high_mel = mel_scale.hz_to_mel(high_freq);

  float delta_mel = (high_mel - low_mel) / (nfilter + 1);
  assert(nfilter > 0);
  std::vector<float> mel_points(nfilter+2, 0.0f);
  mel_points[0] = mel_scale.hz_to_mel(low_freq);
  for (int i = 1; i < nfilter+1; i++) {
    mel_points[i] = mel_points[i-1] + delta_mel;
  }
  mel_points[nfilter+1] = mel_scale.hz_to_mel(high_freq);

  std::vector<float> fftfreqs(nfft/2+1, 0.0f);
  for (int i = 0; i < nfft/2+1; i++) {
    fftfreqs[i] = i * sample_rate / nfft;
  }

  std::vector<float> freq_grid(mel_points.size(), 0.0f);
  int i = 0;
  freq_grid[0] = low_freq;
  for (int i = 1; i < nfilter+1; i++) {
    freq_grid[i] = mel_scale.mel_to_hz(mel_points[i]);
  }
  freq_grid[nfilter+1] = high_freq;

  for (int j = 0; j < nfilter; j++) {
    auto &fbank = fbanks[j];
    fbank.resize(nfft/2+1, 0.0f);
    for (int i = 0; i < nfft/2+1; i++) {
      auto f = fftfreqs[i];
      if (f < low_freq || f > high_freq) {
        fbank[i] = 0.0f;
      } else {
        auto upper = (f - freq_grid[j]) / (freq_grid[j+1] - freq_grid[j]);
        auto lower = (freq_grid[j+2] - f) / (freq_grid[j+2] - freq_grid[j+1]);
        fbank[i] = std::max(0.0f, std::min(upper, lower));
      }
    }
  }

  for (int j = 0; j < nfilter; j++) {
    LOG_LINE << "Filter " << j << " :";
    auto &fbank = fbanks[j];
    for (int i = 0; i < static_cast<int>(fbank.size()); i++) {
      LOG_LINE << " " << fbank[i];
    }
    LOG_LINE << std::endl;
  }
  return fbanks;
}

TEST_P(MelScaleCpuTest, MelScaleCpuTest) {
  using T = float;
  HtkMelScale<float> mel_scale;
  constexpr int Dims = 2;

  auto shape = in_view_.shape;

  int axis = 0;
  int nfft = (shape[axis]-1)*2;
  int nwin = shape[axis+1];

  auto out_shape = in_view_.shape;
  out_shape[axis] = nfilter_;
  auto out_size = volume(out_shape);

  T mel_low = mel_scale.hz_to_mel(freq_low_);
  T mel_high = mel_scale.hz_to_mel(freq_high_);

  T mel_delta = (mel_high - mel_low) / (nfilter_ + 1);
  T mel = mel_low;
  LOG_LINE << "Mel frequency grid (Hz):";
  for (int i = 0; i < nfilter_+1; i++, mel += mel_delta) {
    LOG_LINE << " " << mel_scale.mel_to_hz(mel);
  }
  LOG_LINE << " " << mel_scale.mel_to_hz(mel_high) << "\n";

  LOG_LINE << "FFT bin frequencies (Hz):";
  for (int k = 0; k < nfft / 2 + 1; k++) {
    LOG_LINE << " " << (k * sample_rate_ / nfft);
  }
  LOG_LINE << "\n";

  auto fbanks = ReferenceFilterBanks(nfilter_, nfft, sample_rate_, freq_low_, freq_high_);
  std::vector<T> expected_out(out_size, 0.0f);
  auto expected_out_view = OutTensorCPU<T, Dims>(expected_out.data(), out_shape.to_static<Dims>());
  for (int j = 0; j < nfilter_; j++) {
    for (int t = 0; t < nwin; t++) {
      auto &out_val = expected_out_view.data[j*nwin+t];
      for (int i = 0; i < nfft/2+1; i++) {
        out_val += fbanks[j][i] * in_view_.data[i*nwin+t];
      }
    }
  }

  KernelContext ctx;
  kernels::audio::MelFilterBankArgs args;
  args.axis = Dims - 2;
  args.nfft = nfft;
  args.nfilter = nfilter_;
  args.sample_rate = sample_rate_;
  args.freq_low = freq_low_;
  args.freq_high = freq_high_;
  args.mel_formula = MelScaleFormula::HTK;
  args.normalize = false;

  kernels::audio::MelFilterBankCpu<T, Dims> kernel;
  auto req = kernel.Setup(ctx, in_view_, args);

  ASSERT_EQ(out_shape, req.output_shapes[0][0]);
  std::vector<T> out(out_size, 0.0f);
  auto out_view = OutTensorCPU<T, Dims>(out.data(), out_shape.to_static<Dims>());

  kernel.Run(ctx, out_view, in_view_, args);

  LOG_LINE << "in:\n";
  print_data(in_view_);

  LOG_LINE << "expected out:\n";
  print_data(expected_out_view);

  LOG_LINE << "out:\n";
  print_data(out_view);

  for (int idx = 0; idx < volume(out_view.shape); idx++) {
    ASSERT_NEAR(expected_out_view.data[idx], out_view.data[idx], 1e-4) <<
      "Output data doesn't match reference (idx=" << idx << ")";
  }
}

INSTANTIATE_TEST_SUITE_P(MelScaleCpuTest, MelScaleCpuTest, testing::Combine(
    testing::Values(std::array<int64_t, 2>{17, 1},
                    std::array<int64_t, 2>{513, 111}),  // shape
    testing::Values(4, 8),  // nfilter
    testing::Values(16000.0f),  // sample rate
    testing::Values(0.0f, 1000.0f),  // fmin
    testing::Values(5000.0f, 8000.0f)));  // fmax

}  // namespace test
}  // namespace audio
}  // namespace kernels
}  // namespace dali
