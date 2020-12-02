#include "base/kaldi-common.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"
#include "feat/pitch-functions.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"
 

using namespace kaldi;
using namespace kaldi::nnet3;
typedef kaldi::int32 int32;
typedef kaldi::int64 int64;

void append_mfcc_pitch(const Matrix<BaseFloat> mfcc_features,
                       const Matrix<BaseFloat> pitch_features,
                       Matrix<BaseFloat> *out){

      int32 mfcc_frames,
            pitch_frames,
            mfcc_dim,
            pitch_dim,
            tot_dim;

      mfcc_frames = mfcc_features.NumRows();
      mfcc_dim = mfcc_features.NumCols();

      pitch_frames = pitch_features.NumRows();
      pitch_dim = pitch_features.NumCols();

      if (mfcc_frames != pitch_frames) {
          KALDI_VLOG(2) << "Length mismatch " << mfcc_frames << " vs. " << pitch_frames;
      }

      tot_dim = mfcc_dim + pitch_dim;
      out->Resize(mfcc_frames, tot_dim);
//      std::cout << mfcc_features.Range(0, mfcc_frames, 0, mfcc_dim) << std::endl;
//      std::cout << pitch_features.Range(0, pitch_frames, 0, pitch_dim)  << std::endl;
      out->Range(0, mfcc_frames, 0, mfcc_dim).CopyFromMat(
              mfcc_features.Range(0, mfcc_frames, 0, mfcc_dim));
      out->Range(0, pitch_frames, mfcc_dim, pitch_dim).CopyFromMat(
              pitch_features.Range(0, pitch_frames, 0, pitch_dim));
//      std::cout << out << std::endl;
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Create MFCC and Pitch feature files.\n"
        "Usage: compute-mfcc-pitch_feats [options...] <wav-rspecifier> "
        "<feats-wspecifier>\n";

    // Construct all the global objects.
    ParseOptions po_mfcc(usage);
    ParseOptions po_pitch(usage);
    MfccOptions mfcc_opts;

    PitchExtractionOptions pitch_opts;
    ProcessPitchOptions process_opts;

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

    // Register the MFCC option struct.
    mfcc_opts.Register(&po_mfcc);
    pitch_opts.Register(&po_pitch);
    process_opts.Register(&po_pitch);

    po_mfcc.Read(argc, argv);

    if (po_mfcc.NumArgs() != 2) {
      po_mfcc.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po_mfcc.GetArg(1);
    std::string mfcc_pitch_wspecifier = po_mfcc.GetArg(2);
//    std::string mfcc_wspecifier = po_mfcc.GetArg(2);
//    std::string pitch_wspecifier = po_mfcc.GetArg(3);

    Mfcc mfcc(mfcc_opts);
    Nnet raw_nnet;

    if (utt2spk_rspecifier != "" && vtln_map_rspecifier == "")
      KALDI_ERR << ("The --utt2spk option is only needed if "
                    "the --vtln-map option is used.");
    RandomAccessBaseFloatReaderMapped vtln_map_reader(vtln_map_rspecifier,
                                                      utt2spk_rspecifier);

    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
//    BaseFloatMatrixWriter mfcc_writer;  // typedef to TableWriter<something>.
//    BaseFloatMatrixWriter pitch_writer;  // typedef to TableWriter<something>.
///////    BaseFloatMatrixWriter feats_writer;
    CompressedMatrixWriter feats_writer;

//    if (!mfcc_writer.Open(mfcc_wspecifier))
//        KALDI_ERR << "Could not initialize output with wspecifier "
//                  << mfcc_wspecifier;

//    if (!pitch_writer.Open(pitch_wspecifier))
//        KALDI_ERR << "Could not initialize output with wspecifier "
//                  << pitch_wspecifier;
//
    if (!feats_writer.Open(mfcc_pitch_wspecifier))
        KALDI_ERR << "Could not initialize output with wspecifier "
                  << mfcc_pitch_wspecifier;

    DoubleWriter utt2dur_writer(utt2dur_wspecifier);

    int32 num_utts = 0, num_success = 0;
    for (; !reader.Done(); reader.Next()) {
      num_utts++;
      std::string utt = reader.Key();
      const WaveData &wave_data = reader.Value();
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

      } catch (...) {
        KALDI_WARN << "Failed to compute features for utterance " << utt;
        continue;
      }

//      mfcc_writer.Write(utt, mfcc_features);
//      pitch_writer.Write(utt, pitch_features);
      
////////      feats_writer.Write(utt, output);
      feats_writer.Write(utt, CompressedMatrix(output, compression_method));

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
