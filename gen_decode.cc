// featbin/compute-mfcc-feats.cc

// Copyright 2009-2012  Microsoft Corporation
//                      Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "feat/pitch-functions.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "ivector/voice-activity-detection.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"
#include "base/timer.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Create MFCC feature files.\n"
        "Usage:  compute-mfcc-feats [options...] <wav-rspecifier> "
        "<feats-wspecifier>\n";

    // Construct all the global objects.
    ParseOptions po(usage);
    MfccOptions mfcc_opts;
    VadEnergyOptions vad_opts;
    DeltaFeaturesOptions delta_opts;
    SlidingWindowCmnOptions cmn_opts;
    NnetSimpleComputationOptions nnet_opts;

    PitchExtractionOptions pitch_opts;
    ProcessPitchOptions process_opts;
    pitch_opts.Register(&po);
    process_opts.Register(&po);

    // Register the MFCC option struct.
    mfcc_opts.Register(&po);
    vad_opts.Register(&po);
    delta_opts.Register(&po);
    cmn_opts.Register(&po);
    nnet_opts.Register(&po);

    CuDevice::RegisterDeviceOptions(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 8) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1);
    std::string feats_wspecifier = po.GetArg(2);
    std::string vad_wspecifier = po.GetArg(3);
    std::string delta_wspecifier = po.GetArg(4);
    std::string cmn_wspecifier = po.GetArg(5);
    std::string voiced_frames_wspecifier = po.GetArg(6);
    std::string nnet_rxfilename = po.GetArg(7);
    std::string matrix_wspecifier = po.GetArg(8);

    // Define defaults for global options.
    bool subtract_mean = false;
    BaseFloat vtln_warp = 1.0;
    std::string vtln_map_rspecifier;
    std::string utt2spk_rspecifier;
    int32 channel = -1;
    BaseFloat min_duration = 0.0;
    std::string output_format = "kaldi";
    std::string utt2dur_wspecifier;

    //VAD options.
    bool omit_unvoiced_utts = false;
    int32 num_unvoiced = 0;
    int32 num_done = 0;
    double tot_length = 0.0, tot_decision = 0.0;

    //nnet3 options.
    Timer timer;
    nnet_opts.acoustic_scale = 1.0; // by default do no scaling.

    bool apply_exp = true, use_priors = false;
    std::string use_gpu = "no";
    std::string ivector_rspecifier,
                online_ivector_rspecifier;

    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("use-priors", &use_priors, "If true, subtract the logs of the "
                "priors stored with the model (in this case, "
                "a .mdl file is expected as input).");

    Nnet raw_nnet;
    AmNnetSimple am_nnet;
    if (use_priors) {
      bool binary;
      TransitionModel trans_model;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    } else {
      ReadKaldiObject(nnet_rxfilename, &raw_nnet);
    }

    Nnet &nnet = raw_nnet;
    SetBatchnormTestMode(true, &nnet);
    SetDropoutTestMode(true, &nnet);
    CollapseModel(CollapseModelConfig(), &nnet);
    Vector<BaseFloat> priors;
    CachingOptimizingCompiler compiler(nnet, nnet_opts.optimize_config);
    int32 online_ivector_period = 0;

    CuDevice::Instantiate().SelectGpuId(use_gpu);


    Mfcc mfcc(mfcc_opts);

    int32 compression_method_in = 1;
    CompressionMethod compression_method = static_cast<CompressionMethod>(
        compression_method_in);

    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    CompressedMatrixWriter kaldi_writer(feats_wspecifier);
    BaseFloatVectorWriter vad_writer(vad_wspecifier);
    BaseFloatMatrixWriter feat_writer(delta_wspecifier);
    BaseFloatMatrixWriter cmn_writer(cmn_wspecifier);
    BaseFloatMatrixWriter voiced_frames_writer(voiced_frames_wspecifier);
    BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);

    DoubleWriter utt2dur_writer(utt2dur_wspecifier);

    int32 num_utts = 0, num_success = 0;
    for (; !reader.Done(); reader.Next()) {
      num_utts++;
      std::string utt = reader.Key();
      const WaveData &wave_data = reader.Value();
      const Vector<BaseFloat> *ivector = NULL;
      const Matrix<BaseFloat> *online_ivectors = NULL;
      if (wave_data.Duration() < min_duration) {
        KALDI_WARN << "File: " << utt << " is too short ("
                   << wave_data.Duration() << " sec): producing no output.";
        continue;
      }
      int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
      {  // This block works out the channel (0=left, 1=right...)
        KALDI_ASSERT(num_chan > 0);  // should have been caught in
        // reading code if no channels.
        if (channel == -1) {
          this_chan = 0;
          if (num_chan != 1)
            KALDI_WARN << "Channel not specified but you have data with "
                       << num_chan  << " channels; defaulting to zero";
        } else {
          if (this_chan >= num_chan) {
            KALDI_WARN << "File with id " << utt << " has "
                       << num_chan << " channels but you specified channel "
                       << channel << ", producing no output.";
            continue;
          }
        }
      }
      BaseFloat vtln_warp_local;  // Work out VTLN warp factor.
      vtln_warp_local = vtln_warp;

      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
      Matrix<BaseFloat> features;
      Matrix<BaseFloat> pitch_features;
      Matrix<BaseFloat> mfcc_features;
      try {

        //compute MFCC and pitch features
        mfcc.ComputeFeatures(waveform, wave_data.SampFreq(),
                             vtln_warp_local, &features);

        ComputeAndProcessKaldiPitch(pitch_opts, process_opts,
                                    waveform, &pitch_features);
      } catch (...) {
        KALDI_WARN << "Failed to compute features for utterance " << utt;
        continue;
      }
      if (subtract_mean) {
        Vector<BaseFloat> mean(features.NumCols());
        mean.AddRowSumMat(1.0, features);
        mean.Scale(1.0 / features.NumRows());
        for (int32 i = 0; i < features.NumRows(); i++)
          features.Row(i).AddVec(-1.0, mean);
      }

      if (output_format == "kaldi") {
        CompressedMatrix tmp = CompressedMatrix(features, compression_method);
        kaldi_writer.Write(utt, tmp);

        mfcc_features = features;
        // comupute VAD 
        Vector<BaseFloat> vad_result(mfcc_features.NumRows());
        ComputeVadEnergy(vad_opts, mfcc_features, &vad_result);

        double sum = vad_result.Sum();
        if (sum == 0.0) {
          KALDI_WARN << "No frames were judged voiced for utterance " << utt;
          num_unvoiced++;
        } else {
          num_done++;
        }
        tot_decision += vad_result.Sum();
        tot_length += vad_result.Dim();
  
        if (!(omit_unvoiced_utts && sum == 0)) {
          vad_writer.Write(utt, vad_result);
        }

        // compute delta features
        Matrix<BaseFloat> new_feats;
        ComputeDeltas(delta_opts, mfcc_features, &new_feats);
        feat_writer.Write(utt, new_feats);

        // compute cmn features
        Matrix<BaseFloat> cmvn_feat(new_feats.NumRows(),
                                  new_feats.NumCols(), kUndefined);

        SlidingWindowCmn(cmn_opts, new_feats, &cmvn_feat);
        cmn_writer.Write(utt, cmvn_feat);

        // select voiced frames by VAD
        const Vector<BaseFloat> &voiced = vad_result;
        int32 dim = 0;
        for (int32 i = 0; i < voiced.Dim(); i++)
            if (voiced(i) != 0.0)
                dim++;

        Matrix<BaseFloat> voiced_feat(dim, cmvn_feat.NumCols());
        int32 index = 0;
        for (int32 i = 0; i < cmvn_feat.NumRows(); i++) {
            if (voiced(i) != 0.0) {
                KALDI_ASSERT(voiced(i) == 1.0); // should be zero or one.
                voiced_feat.Row(index).CopyFromVec(cmvn_feat.Row(i));
                index++;
            }
        }
        voiced_frames_writer.Write(utt, voiced_feat);

        // compute nnet result 
        DecodableNnetSimple nnet_computer(
            nnet_opts, nnet, priors,
            voiced_feat, &compiler,
            ivector, online_ivectors,
            online_ivector_period);

        Matrix<BaseFloat> matrix(nnet_computer.NumFrames(),
                               nnet_computer.OutputDim());

        std::cout << nnet_computer.NumFrames() << std::endl;

        if (apply_exp)
            matrix.ApplyExp();

        matrix_writer.Write(utt, matrix);

      } 

      if (utt2dur_writer.IsOpen()) {
        utt2dur_writer.Write(utt, wave_data.Duration());
      }

      if (num_utts % 10 == 0)
        KALDI_LOG << "Processed " << num_utts << " utterances";
      KALDI_VLOG(2) << "Processed features for key " << utt;
      num_success++;
    }
    KALDI_LOG << " Done " << num_success << " out of " << num_utts
              << " utterances.";
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
