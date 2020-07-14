#include "base/kaldi-common.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"
#include "feat/pitch-functions.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"
 

using namespace kaldi;
void append_mfcc_pitch(const Matrix<BaseFloat> mfcc_features,
                       const Matrix<BaseFloat> pitch_features,
                       Matrix<BaseFloat> *out){

      int32 mfcc_frames,
            pitch_frames,
            min_frames,
            max_frames,
            mfcc_dim,
            pitch_dim,
            tot_dim;

      mfcc_frames = mfcc_features.NumRows();
      mfcc_dim = mfcc_features.NumCols();

      pitch_frames = pitch_features.NumRows();
      pitch_dim = pitch_features.NumCols();

      min_frames = mfcc_frames;
      max_frames = mfcc_frames;
      if (pitch_frames < mfcc_frames) {
          min_frames = pitch_frames;
      }
      if (pitch_frames > mfcc_frames) {
          max_frames = pitch_frames;
      }

      if (min_frames != max_frames) {
          KALDI_VLOG(2) << "Length mismatch " << mfcc_frames << " vs. " << pitch_frames;
          if (max_frames - min_frames > 2){
              KALDI_WARN << "frames have too large different!!!" << max_frames << "   "<< min_frames ;
              exit(-1);
          }
      }

      tot_dim = mfcc_dim + pitch_dim;
      out->Resize(min_frames, tot_dim);
      out->Range(0, min_frames, 0, mfcc_dim).CopyFromMat(
              mfcc_features.Range(0, min_frames, 0, mfcc_dim));
      out->Range(0, min_frames, mfcc_dim, pitch_dim).CopyFromMat(
              pitch_features.Range(0, min_frames, 0, pitch_dim));
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get vad by NN.\n"
        "Usage: wav2vad  [options...] <model-path> <wav-rspecifier> <vad-wspecifier>\n"
        "e.g.: wav2vad final.raw scp:wav.scp ark,t:vad.txt\n";


    // Construct all the global objects.
    ParseOptions po(usage);

    MfccOptions mfcc_opts;
    PitchExtractionOptions pitch_opts;
    ProcessPitchOptions process_opts;
    NnetSimpleComputationOptions vad_opts;

    Timer timer;

    // Define defaults for global options.
    BaseFloat vtln_warp = 1.0;
    std::string vtln_map_rspecifier;
    std::string utt2spk_rspecifier;
    int32 channel = -1;
    BaseFloat min_duration = 0.0;
    std::string output_format = "kaldi";
    std::string utt2dur_wspecifier;
    int32 compression_method_in = 1;
    mfcc_opts.use_energy = false;

    vad_opts.acoustic_scale = 1.0;
    vad_opts.frames_per_chunk = 150;
    bool use_priors = false;
    std::string use_gpu = "no";

    std::string ivector_rspecifier,
                online_ivector_rspecifier;

    int32 online_ivector_period = 0;

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
                wav_rspecifier = po.GetArg(2),
                vad_wspecifier = po.GetArg(3);

    Mfcc mfcc(mfcc_opts);
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
 //   ReadKaldiObject(nnet_rxfilename, &raw_nnet);

    Nnet &nnet = raw_nnet;
    SetBatchnormTestMode(true, &nnet);
    SetDropoutTestMode(true, &nnet);
    CollapseModel(CollapseModelConfig(), &nnet);

    Vector<BaseFloat> priors;

    RandomAccessBaseFloatMatrixReader online_ivector_reader(
        online_ivector_rspecifier);
    RandomAccessBaseFloatVectorReaderMapped ivector_reader(
        ivector_rspecifier, utt2spk_rspecifier);

    CachingOptimizingCompiler compiler(nnet, vad_opts.optimize_config);
//    BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);
    BaseFloatVectorWriter vad_writer(vad_wspecifier);

    if (utt2spk_rspecifier != "" && vtln_map_rspecifier == "")
      KALDI_ERR << ("The --utt2spk option is only needed if "
                    "the --vtln-map option is used.");
    RandomAccessBaseFloatReaderMapped vtln_map_reader(vtln_map_rspecifier,
                                                      utt2spk_rspecifier);

    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    CompressedMatrixWriter feats_writer;

    DoubleWriter utt2dur_writer(utt2dur_wspecifier);

    int32 num_utts = 0, num_success = 0;
    int64 frame_count = 0;
    for (; !reader.Done(); reader.Next()) {
      num_utts++;
      std::string utt = reader.Key();
      const WaveData &wave_data = reader.Value();
      const Matrix<BaseFloat> *online_ivectors = NULL;
      const Vector<BaseFloat> *ivector = NULL;

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
      Matrix<BaseFloat> mfcc_features;
      Matrix<BaseFloat> pitch_features;

      Matrix<BaseFloat> output;

      CompressionMethod compression_method = static_cast<CompressionMethod>(
        compression_method_in);
      try {
        mfcc.ComputeFeatures(waveform, wave_data.SampFreq(),
                             vtln_warp_local, &mfcc_features);

//        std::cout << mfcc_features << std::endl;
        ComputeAndProcessKaldiPitch(pitch_opts, process_opts,
                                    waveform, &pitch_features);

        append_mfcc_pitch(mfcc_features, pitch_features, &output);
//        std::cout << output << std::endl;

        Matrix<BaseFloat> features(output.NumRows(), output.NumCols());
        CompressedMatrix mat(output, compression_method);
        mat.CopyToMat(&features);
//        std::cout << features << std::endl;

        DecodableNnetSimple nnet_computer(
            vad_opts, nnet, priors,
            features, &compiler,
            ivector, online_ivectors,
            online_ivector_period);

        Matrix<BaseFloat> matrix(nnet_computer.NumFrames(),
                               nnet_computer.OutputDim());

        for (int32 t = 0; t < nnet_computer.NumFrames(); t++) {
//          std::cout << t << std::endl;
//          std::cout << matrix << std::endl;
          SubVector<BaseFloat> row(matrix, t);
          nnet_computer.GetOutputForFrame(t, &row);
        }

        matrix.ApplyExp();
//        std::cout << matrix << std::endl;

        Vector<BaseFloat> vad_result(nnet_computer.NumFrames());
        for(int i=0; i<nnet_computer.NumFrames(); i++) {
           if(matrix(i,0) > matrix(i,2)) {
             vad_result(i) = 0; }
           else {
             vad_result(i) = 1; 
           }
        }
//       std::cout << vad_result << std::endl;
//       matrix_writer.Write(utt, matrix);
       vad_writer.Write(utt, vad_result);
       frame_count += features.NumRows();
      } catch (...) {
        KALDI_WARN << "Failed to compute features for utterance " << utt;
        continue;
      }


      if (utt2dur_writer.IsOpen()) {
        utt2dur_writer.Write(utt, wave_data.Duration());
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
    KALDI_LOG << " Done " << num_success << " out of " << num_utts
              << " utterances.";
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances ";

    if (num_success != 0) return 0;
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
