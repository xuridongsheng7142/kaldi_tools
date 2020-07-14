#include "base/kaldi-common.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"
#include "ivector/voice-activity-detection.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"

using namespace kaldi;
using namespace kaldi::nnet3;
typedef kaldi::int32 int32;
typedef kaldi::int64 int64;

void get_mfcc(const WaveData wave_data,
              Matrix<BaseFloat> *features){

    MfccOptions mfcc_opts;
    mfcc_opts.frame_opts.samp_freq = 16000;
    mfcc_opts.mel_opts.num_bins = 40;
    mfcc_opts.num_ceps = 20;
    Mfcc mfcc(mfcc_opts);
    int32 this_chan = 0;
    BaseFloat vtln_warp_local = 1.0;
    SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
    mfcc.ComputeFeatures(waveform, wave_data.SampFreq(),
                             vtln_warp_local, features);
}

void get_Compressed_mfcc(Matrix<BaseFloat> features, 
                        Matrix<BaseFloat> *mfcc_feats){

    int32 compression_method_in = 1;
    CompressionMethod compression_method = static_cast<CompressionMethod>(
      compression_method_in);
    CompressedMatrix mat(features, compression_method);
    mat.CopyToMat(mfcc_feats);
}

void get_vad(Matrix<BaseFloat> mfcc_feats,
             Vector<BaseFloat> *vad_result){

    VadEnergyOptions vad_opts;
    vad_opts.vad_energy_threshold = 5.5;
    vad_opts.vad_energy_mean_scale = 0.5;
    ComputeVadEnergy(vad_opts, mfcc_feats, vad_result);
}

void get_delta(Matrix<BaseFloat> mfcc_feats,
              Matrix<BaseFloat> *new_feats){

    DeltaFeaturesOptions delta_opts;
    delta_opts.order = 2;
    delta_opts.window = 3;

    ComputeDeltas(delta_opts, mfcc_feats, new_feats);
}

void get_slid_win_cmn(Matrix<BaseFloat> new_feats,
                     Matrix<BaseFloat> *cmvn_feat){
    SlidingWindowCmnOptions slid_win_opts;
    slid_win_opts.normalize_variance = false;
    slid_win_opts.center = true;
    slid_win_opts.cmn_window = 300;
    SlidingWindowCmn(slid_win_opts, new_feats, cmvn_feat);
}


int main(int argc, char *argv[]) {
  try {
    const char *usage =
        "Get gender prediction.\n"
        "Usage: wav2gen [options...] <model-path> <wav-rspecifier> <gen-pre>"
        "<feats-wspecifier>\n";

    ParseOptions po(usage);
    Timer timer;
    NnetSimpleComputationOptions nnet_opts;
    Nnet raw_nnet;
    Nnet &nnet = raw_nnet;
    Vector<BaseFloat> priors;
    CachingOptimizingCompiler compiler(nnet, nnet_opts.optimize_config);

    BaseFloat min_duration = 0.0;
    const Vector<BaseFloat> *ivector = NULL;
    const Matrix<BaseFloat> *online_ivectors = NULL;
    int32 online_ivector_period = 0;
    std::string use_gpu = "no";
    nnet_opts.acoustic_scale = 1.0;
    nnet_opts.frames_per_chunk = 150;
    nnet_opts.extra_left_context = 0;
    nnet_opts.extra_right_context = 0;
    nnet_opts.extra_left_context_initial = -1;
    nnet_opts.extra_right_context_final = -1;

#if HAVE_CUDA==1
    CuDevice::RegisterDeviceOptions(&po);
#endif

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1);
    std::string wav_rspecifier = po.GetArg(2);
    std::string output_wspecifier = po.GetArg(3);

    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    BaseFloatVectorWriter kaldi_writer;  // typedef to TableWriter<something>.

    if (!kaldi_writer.Open(output_wspecifier))
        KALDI_ERR << "Could not initialize output with wspecifier "
                << output_wspecifier;

    int32 num_utts = 0, num_success = 0, num_err = 0;
    int64 frame_count = 0;
    for (; !reader.Done(); reader.Next()) {
      num_utts++;
      std::string utt = reader.Key();
      const WaveData &wave_data = reader.Value();
      if (wave_data.Duration() < min_duration) {
        KALDI_WARN << "File: " << utt << " is too short ("
                   << wave_data.Duration() << " sec): producing no output.";
        continue;
      }

      Matrix<BaseFloat> features;
      Matrix<BaseFloat> new_feats;

      try {
        get_mfcc(wave_data, &features);

        Matrix<BaseFloat> mfcc_feats(features.NumRows(), features.NumCols());
        get_Compressed_mfcc(features, &mfcc_feats);

        Vector<BaseFloat> vad_result(mfcc_feats.NumRows());
        get_vad(mfcc_feats, &vad_result);
        
        get_delta(mfcc_feats, &new_feats);

        Matrix<BaseFloat> cmvn_feat(new_feats.NumRows(), new_feats.NumCols(), kUndefined);
        get_slid_win_cmn(new_feats, &cmvn_feat);

        if (cmvn_feat.NumRows() != vad_result.Dim()) {
          KALDI_WARN << "Mismatch in number for frames " << cmvn_feat.NumRows()
                     << " for features and VAD " << vad_result.Dim()
                     << ", for utterance " << utt;
          num_err++;
          continue;
        }
        if (vad_result.Sum() == 0.0) {
          KALDI_WARN << "No features were judged as voiced for utterance "
                     << utt;
          num_err++;
          continue;
        }
        int32 dim = 0;
        for (int32 i = 0; i < vad_result.Dim(); i++)
          if (vad_result(i) != 0.0)
            dim++;
        Matrix<BaseFloat> voiced_feat(dim, cmvn_feat.NumCols());
        int32 index = 0;
        for (int32 i = 0; i < cmvn_feat.NumRows(); i++) {
          if (vad_result(i) != 0.0) {
            KALDI_ASSERT(vad_result(i) == 1.0); // should be zero or one.
            voiced_feat.Row(index).CopyFromVec(cmvn_feat.Row(i));
            index++;
          }
        }
        KALDI_ASSERT(index == dim);

//        std::cout << voiced_feat << std::endl;

        ReadKaldiObject(nnet_rxfilename, &raw_nnet);
        DecodableNnetSimple nnet_computer(
            nnet_opts, nnet, priors,
            voiced_feat, &compiler,
            ivector, online_ivectors,
            online_ivector_period);
    
        Matrix<BaseFloat> matrix(nnet_computer.NumFrames(),
                                 nnet_computer.OutputDim());
        for (int32 t = 0; t < nnet_computer.NumFrames(); t++) {
          SubVector<BaseFloat> row(matrix, t);
          nnet_computer.GetOutputForFrame(t, &row);
        }
        matrix.ApplyExp();
        frame_count += matrix.NumRows();

        Vector<BaseFloat> gen_pre(2);
        float M_score = 0.0;
        float F_score = 0.0;
        for(int i=0; i<nnet_computer.NumFrames(); i++) {
            M_score += matrix(i,0);
            F_score += matrix(i,1);
        }
        float M_ave = M_score / nnet_computer.NumFrames();
        float F_ave = F_score / nnet_computer.NumFrames();
        if(M_ave >= F_ave) {
            gen_pre(0) = 0;
            gen_pre(1) = M_ave - F_ave; }
//            std::cout << utt << " M " << M_ave - F_ave << "\n"; 
        else {
            gen_pre(0) = 1;
            gen_pre(1) = F_ave - M_ave; }
//            std::cout << utt << " F " << F_ave - M_ave << "\n";
        kaldi_writer.Write(utt, gen_pre);

      } catch (...) {
        KALDI_WARN << "Failed to compute features for utterance " << utt;
        continue;
      }

      if (num_utts % 10 == 0)
        KALDI_LOG << "Processed " << num_utts << " utterances";
      KALDI_VLOG(2) << "Processed features for key " << utt;
      num_success++;
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances ";

    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
