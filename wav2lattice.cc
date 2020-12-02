#include "feat/wave-reader.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"
#include "lat/word-align-lattice-lexicon.h"
#include "kws/kaldi-kws.h"
#include "kws/kws-functions.h"
#include "fstext/epsilon-property.h"
#include "fstext/kaldi-fst-io.h"
#include "util/common-utils.h"

namespace kaldi {

typedef KwsLexicographicArc Arc;
typedef Arc::Weight Weight;
typedef Arc::StateId StateId;

// encode ilabel, olabel pair as a single 64bit (output) symbol
uint64 EncodeLabel(StateId ilabel, StateId olabel) {
  return (static_cast<int64>(olabel) << 32) + static_cast<int64>(ilabel);
}

StateId DecodeLabelUid(uint64 osymbol) {
  return static_cast<StateId>(osymbol >> 32);
}

class VectorFstToKwsLexicographicFstMapper {
 public:
  typedef fst::StdArc FromArc;
  typedef FromArc::Weight FromWeight;
  typedef KwsLexicographicArc ToArc;
  typedef KwsLexicographicWeight ToWeight;

  VectorFstToKwsLexicographicFstMapper() {}

  ToArc operator()(const FromArc &arc) const {
    return ToArc(arc.ilabel,
                 arc.olabel,
                 (arc.weight == FromWeight::Zero() ?
                  ToWeight::Zero() :
                  ToWeight(arc.weight.Value(),
                           StdLStdWeight::One())),
                 arc.nextstate);
  }

  fst::MapFinalAction FinalAction() const {
    return fst::MAP_NO_SUPERFINAL;
  }

  fst::MapSymbolsAction InputSymbolsAction() const {
    return fst::MAP_COPY_SYMBOLS;
  }

  fst::MapSymbolsAction OutputSymbolsAction() const {
    return fst::MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const { return props; }
};

struct ActivePath {
  std::vector<KwsLexicographicArc::Label> path;
  KwsLexicographicArc::Weight weight;
  KwsLexicographicArc::Label last;
};

bool GenerateActivePaths(const KwsLexicographicFst &proxy,
                       std::vector<ActivePath> *paths,
                       KwsLexicographicFst::StateId cur_state,
                       std::vector<KwsLexicographicArc::Label> cur_path,
                       KwsLexicographicArc::Weight cur_weight) {
  for (fst::ArcIterator<KwsLexicographicFst> aiter(proxy, cur_state);
       !aiter.Done(); aiter.Next()) {
    const Arc &arc = aiter.Value();
    Weight temp_weight = Times(arc.weight, cur_weight);

    cur_path.push_back(arc.ilabel);

    if ( arc.olabel != 0 ) {
      ActivePath path;
      path.path = cur_path;
      path.weight = temp_weight;
      path.last = arc.olabel;
      paths->push_back(path);
    } else {
      GenerateActivePaths(proxy, paths,
                        arc.nextstate, cur_path, temp_weight);
    }
    cur_path.pop_back();
  }

  return true;
}

typedef kaldi::TableWriter< kaldi::BasicVectorHolder<double> >
                                                        VectorOfDoublesWriter;

void OutputDetailedStatistics(const std::string &kwid,
                        const kaldi::KwsLexicographicFst &keyword,
                        const unordered_map<uint32, uint64> &label_decoder,
                        VectorOfDoublesWriter *output ) {
  std::vector<kaldi::ActivePath> paths;

  if (keyword.Start() == fst::kNoStateId)
    return;

  kaldi::GenerateActivePaths(keyword, &paths, keyword.Start(),
                  std::vector<kaldi::KwsLexicographicArc::Label>(),
                  kaldi::KwsLexicographicArc::Weight::One());

  for (int i = 0; i < paths.size(); ++i) {
    std::vector<double> out;
    double score;
    int32 tbeg, tend, uid;

    uint64 osymbol = label_decoder.find(paths[i].last)->second;
    uid = kaldi::DecodeLabelUid(osymbol);
    tbeg = paths[i].weight.Value2().Value1().Value();
    tend = paths[i].weight.Value2().Value2().Value();
    score = paths[i].weight.Value1().Value();

    out.push_back(uid);
    out.push_back(tbeg);
    out.push_back(tend);
    out.push_back(score);

    for (int j = 0; j < paths[i].path.size(); ++j) {
      out.push_back(paths[i].path[j]);
    }
    output->Write(kwid, out);
  }
}

void GetDiagnosticsAndPrintOutput(const std::string &utt,
                                  const fst::SymbolTable *word_syms,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like) {
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return;
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  num_frames = alignment.size();
  likelihood = -(weight.Value1() + weight.Value2());
  *tot_num_frames += num_frames;
  *tot_like += likelihood;
  KALDI_VLOG(2) << "Likelihood per frame for utterance " << utt << " is "
                << (likelihood / num_frames) << " over " << num_frames
                << " frames.";

  if (word_syms != NULL) {
    std::cerr << utt << ' ';
    for (size_t i = 0; i < words.size(); i++) {
      std::string s = word_syms->Find(words[i]);
      if (s == "")
        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
      std::cerr << s << ' ';
    }
    std::cerr << std::endl;
  }
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in wav file(s) and simulates online decoding with neural nets\n"
        "(nnet3 setup), with optional iVector-based speaker adaptation and\n"
        "optional endpointing.  Note: some configuration values and inputs are\n"
        "set via config files whose filenames are passed as options\n"
        "\n"
        "Usage: online2-wav-nnet3-latgen-faster [options] <nnet3-in> <fst-in> "
        "<spk2utt-rspecifier> <wav-rspecifier> <lattice-wspecifier>\n"
        "The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
        "you want to decode utterance by utterance.\n";

    ParseOptions po(usage);

    std::string word_syms_rxfilename;

    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_opts;
    OnlineEndpointConfig endpoint_opts;
    WordAlignLatticeLexiconOpts word_align_opts;

    BaseFloat chunk_length_secs = 0.18;
    bool do_endpointing = false;
    bool online = true;

    feature_opts.feature_type = "mfcc";
    feature_opts.add_pitch = true;
    decodable_opts.extra_left_context_initial = 0;
    decodable_opts.frame_subsampling_factor = 3;
    decodable_opts.frames_per_chunk = 50;
    decodable_opts.acoustic_scale = 1.0;
    decoder_opts.beam = 15.0;
    decoder_opts.max_active = 7000;
    decoder_opts.min_active = 200;
    decoder_opts.lattice_beam = 8.0;
    endpoint_opts.silence_phones = "1:2:3:4";

    // scales for lattice
    BaseFloat lm_scale = 1.0;
    BaseFloat acoustic2lm_scale = 0.0;
    BaseFloat lm2acoustic_scale = 0.0;
    BaseFloat inv_acoustic_scale = 7.0;
    BaseFloat acoustic_scale;
    acoustic_scale = 10.0 / inv_acoustic_scale;

    std::vector<std::vector<double> > scale(2);
    scale[0].resize(2);
    scale[1].resize(2);
    scale[0][0] = lm_scale;
    scale[0][1] = acoustic2lm_scale;
    scale[1][0] = lm2acoustic_scale;
    scale[1][1] = acoustic_scale;

    // word_ins_penalty
    BaseFloat word_ins_penalty = 1.0;

    // scores for lattice-to-kws-index
    int32 max_silence_frames = 50;
    bool allow_partial = true;
    BaseFloat max_states_scale = 4;
    int32 max_states = -1;
    max_silence_frames = 0.5 + max_silence_frames / static_cast<float>(decodable_opts.frame_subsampling_factor);

    // kws-search
    double keyword_beam = -1;
    int32 keyword_nbest = -1;
    double negative_tolerance = -0.1;
    int32 n_best = -1;

    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po);
    endpoint_opts.Register(&po);

    po.Read(argc, argv);

   if (po.NumArgs() != 7 && po.NumArgs() != 8) {
      std::cout << "Usage:\n\t wav2lattice <nnet3_rxfilename> <fst_rxfilename> <wav_rspecifier> <align_lexicon_rxfilename> <usymtab_rspecifier> <keyword_rspecifier> <result_wspecifier> (<stats_wspecifier>)\n";
      return 1;
    }

    std::string nnet3_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        wav_rspecifier = po.GetArg(3),
        align_lexicon_rxfilename = po.GetArg(4),
        usymtab_rspecifier = po.GetArg(5),
        keyword_rspecifier = po.GetArg(6),
        result_wspecifier = po.GetArg(7);
    std::string stats_wspecifier = "";
    if (po.NumArgs() != 8) {
        stats_wspecifier = po.GetOptArg(8);
    }

    std::vector<std::vector<int32> > lexicon;
    {
      bool binary_in;
      Input ki(align_lexicon_rxfilename, &binary_in);
      KALDI_ASSERT(!binary_in && "Not expecting binary file for lexicon");
      if (!ReadLexiconForWordAlign(ki.Stream(), &lexicon)) {
        KALDI_ERR << "Error reading alignment lexicon from "
                  << align_lexicon_rxfilename;
      }
    }

    WordAlignLatticeLexiconInfo lexicon_info(lexicon);

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);
    if (!online) {
      feature_info.ivector_extractor_info.use_most_recent_ivector = true;
      feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
      chunk_length_secs = -1.0;
    }

    Matrix<double> global_cmvn_stats;
    if (feature_info.global_cmvn_stats_rxfilename != "")
      ReadKaldiObject(feature_info.global_cmvn_stats_rxfilename,
                      &global_cmvn_stats);

    TransitionModel trans_model;
    nnet3::AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet3_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    // this object contains precomputed stuff that is used by all decodable
    // objects.  It takes a pointer to am_nnet because if it has iVectors it has
    // to modify the nnet to accept iVectors at intervals.
    nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,
                                                        &am_nnet);

    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;

    // We use RandomAccessInt32Reader to read the utterance symtab table.
    RandomAccessInt32Reader usymtab_reader(usymtab_rspecifier);
    SequentialTableReader<VectorFstHolder> keyword_reader(keyword_rspecifier);

    int32 num_done = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;

    SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);

    VectorOfDoublesWriter result_writer(result_wspecifier);
    VectorOfDoublesWriter stats_writer(stats_wspecifier);

    OnlineTimingStats timing_stats;
    KwsLexicographicFst global_index;

    OnlineIvectorExtractorAdaptationState adaptation_state(
        feature_info.ivector_extractor_info);
    OnlineCmvnState cmvn_state(global_cmvn_stats);

    for (; !wav_reader.Done(); wav_reader.Next()) {
        std::string utt = wav_reader.Key();
        const WaveData &wave_data = wav_reader.Value();
        // get the data for channel zero (if the signal is not mono, we only
        // take the first channel).
        SubVector<BaseFloat> data(wave_data.Data(), 0);

        OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
        feature_pipeline.SetAdaptationState(adaptation_state);
        feature_pipeline.SetCmvnState(cmvn_state);

        OnlineSilenceWeighting silence_weighting(
            trans_model,
            feature_info.silence_weighting_config,
            decodable_opts.frame_subsampling_factor);

        SingleUtteranceNnet3Decoder decoder(decoder_opts, trans_model,
                                            decodable_info,
                                            *decode_fst, &feature_pipeline);
        OnlineTimer decoding_timer(utt);

        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length;
        if (chunk_length_secs > 0) {
          chunk_length = int32(samp_freq * chunk_length_secs);
          if (chunk_length == 0) chunk_length = 1;
        } else {
          chunk_length = std::numeric_limits<int32>::max();
        }

        int32 samp_offset = 0;
        std::vector<std::pair<int32, BaseFloat> > delta_weights;

        while (samp_offset < data.Dim()) {
          int32 samp_remaining = data.Dim() - samp_offset;
          int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                         : samp_remaining;

          SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
          feature_pipeline.AcceptWaveform(samp_freq, wave_part);

          samp_offset += num_samp;
          decoding_timer.WaitUntil(samp_offset / samp_freq);
          if (samp_offset == data.Dim()) {
            // no more input. flush out last frames
            feature_pipeline.InputFinished();
          }

          if (silence_weighting.Active() &&
              feature_pipeline.IvectorFeature() != NULL) {
            silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
            silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                              &delta_weights);
            feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
          }

          decoder.AdvanceDecoding();

          if (do_endpointing && decoder.EndpointDetected(endpoint_opts)) {
            break;
          }
        }
        decoder.FinalizeDecoding();

        CompactLattice clat;
        bool end_of_utterance = true;
        decoder.GetLattice(end_of_utterance, &clat);

        GetDiagnosticsAndPrintOutput(utt, word_syms, clat,
                                     &num_frames, &tot_like);

        decoding_timer.OutputStats(&timing_stats);

        // In an application you might avoid updating the adaptation state if
        // you felt the utterance had low confidence.  See lat/confidence.h
        feature_pipeline.GetAdaptationState(&adaptation_state);
        feature_pipeline.GetCmvnState(&cmvn_state);

        ScaleLattice(scale, &clat);
        AddWordInsPenToCompactLattice(word_ins_penalty, &clat);

        CompactLattice aligned_clat;
        bool ok = WordAlignLatticeLexicon(clat, trans_model, lexicon_info, word_align_opts, &aligned_clat);

        if (ok) {
          TopSortCompactLatticeIfNeeded(&aligned_clat);
          // Topologically sort the lattice, if not already sorted.
          uint64 props = aligned_clat.Properties(fst::kFstProperties, false);
          if (!(props & fst::kTopSorted)) {
            if (fst::TopSort(&clat) == false) {
              KALDI_WARN << "Cycles detected in lattice " << utt;
              continue;
            }
          }
          // Get the alignments
          std::vector<int32> state_times;
          CompactLatticeStateTimes(aligned_clat, &state_times);
          bool success = false;
          success = kaldi::ClusterLattice(&aligned_clat, state_times);

          if (true) {
            EnsureEpsilonProperty(&aligned_clat);
            fst::TopSort(&aligned_clat);
            // We have to recompute the state times because they will have changed.
            CompactLatticeStateTimes(aligned_clat, &state_times);
          }

          KwsProductFst factor_transducer;
          int32 utterance_id = usymtab_reader.Value(utt);
          success = kaldi::CreateFactorTransducer(aligned_clat,
                                              state_times,
                                              utterance_id,
                                              &factor_transducer);
          if (!success) {
            KALDI_WARN << "Cannot generate factor transducer for lattice " << utt;
          }

          MaybeDoSanityCheck(factor_transducer);
          RemoveLongSilences(max_silence_frames, state_times, &factor_transducer);
          MaybeDoSanityCheck(factor_transducer);
          KwsLexicographicFst index_transducer;
          DoFactorMerging(&factor_transducer, &index_transducer);
          MaybeDoSanityCheck(index_transducer);
          DoFactorDisambiguation(&index_transducer);
          MaybeDoSanityCheck(index_transducer);
          max_states = static_cast<int32>(max_states_scale * static_cast<BaseFloat>(aligned_clat.NumStates()));
          OptimizeFactorTransducer(&index_transducer, max_states, allow_partial);
          MaybeDoSanityCheck(index_transducer);
          Union(&global_index, index_transducer);

          KALDI_LOG << "Decoded utterance " << utt;
          num_done++;
        }
      
    }

    max_states = -1;
    KwsLexicographicFst ifst = global_index;
    EncodeMapper<KwsLexicographicArc> encoder(kEncodeLabels, ENCODE);
    Encode(&ifst, &encoder);
    try {
      DeterminizeStar(ifst, &global_index, kDelta, NULL, max_states);
    } catch(const std::exception &e) {
      KALDI_WARN << e.what()
                 << " (should affect speed of search but not results)";
      global_index = ifst;
    }
    Minimize(&global_index, static_cast<KwsLexicographicFst*>(NULL), kDelta, true);
    Decode(&global_index, encoder);
//    index_writer.Write("global", global_index);

    int32 label_count = 1;
    unordered_map<uint64, uint32> label_encoder;
    unordered_map<uint32, uint64> label_decoder;
    for (StateIterator<KwsLexicographicFst> siter(global_index);
                                           !siter.Done(); siter.Next()) {
      StateId state_id = siter.Value();
      for (MutableArcIterator<KwsLexicographicFst>
           aiter(&global_index, state_id); !aiter.Done(); aiter.Next()) {
        KwsLexicographicArc arc = aiter.Value();
        // Skip the non-final arcs
        if (global_index.Final(arc.nextstate) == Weight::Zero())
          continue;
        // Encode the input and output label of the final arc, and this is the
        // new output label for this arc; set the input label to <epsilon>
        uint64 osymbol = EncodeLabel(arc.ilabel, arc.olabel);
        arc.ilabel = 0;
        if (label_encoder.find(osymbol) == label_encoder.end()) {
          arc.olabel = label_count;
          label_encoder[osymbol] = label_count;
          label_decoder[label_count] = osymbol;
          label_count++;
        } else {
          arc.olabel = label_encoder[osymbol];
        }
        aiter.SetValue(arc);
      }
    }
    ArcSort(&global_index, fst::ILabelCompare<KwsLexicographicArc>());

    int32 n_done = 0;
    int32 n_fail = 0;
    for (; !keyword_reader.Done(); keyword_reader.Next()) {
      std::string key = keyword_reader.Key();
      VectorFst<StdArc> keyword = keyword_reader.Value();
      keyword_reader.FreeCurrent();

      // Process the case where we have confusion for keywords
      if (keyword_beam != -1) {
        Prune(&keyword, keyword_beam);
      }
      if (keyword_nbest != -1) {
        VectorFst<StdArc> tmp;
        ShortestPath(keyword, &tmp, keyword_nbest, true, true);
        keyword = tmp;
      }

      KwsLexicographicFst keyword_fst;
      KwsLexicographicFst result_fst;
      Map(keyword, &keyword_fst, VectorFstToKwsLexicographicFstMapper());
      Compose(keyword_fst, global_index, &result_fst);

      if (stats_wspecifier != "") {
        KwsLexicographicFst matched_seq(result_fst);
        OutputDetailedStatistics(key,
                                 matched_seq,
                                 label_decoder,
                                 &stats_writer);
      }

      Project(&result_fst, PROJECT_OUTPUT);
      Minimize(&result_fst, (KwsLexicographicFst *) nullptr, kDelta, true);
      ShortestPath(result_fst, &result_fst, n_best);
      RmEpsilon(&result_fst);

      // No result found
      if (result_fst.Start() == kNoStateId)
        continue;

      // Got something here
      double score;
      int32 tbeg, tend, uid;
      for (ArcIterator<KwsLexicographicFst>
           aiter(result_fst, result_fst.Start()); !aiter.Done(); aiter.Next()) {
        const KwsLexicographicArc &arc = aiter.Value();

        // We're expecting a two-state FST
        if (result_fst.Final(arc.nextstate) != Weight::One()) {
          KALDI_WARN << "The resulting FST does not have "
                     << "the expected structure for key " << key;
          n_fail++;
          continue;
        }

        uint64 osymbol = label_decoder[arc.olabel];
        uid = static_cast<int32>(DecodeLabelUid(osymbol));
        tbeg = arc.weight.Value2().Value1().Value();
        tend = arc.weight.Value2().Value2().Value();
        score = arc.weight.Value1().Value();

        if (score < 0) {
          if (score < negative_tolerance) {
            KALDI_WARN << "Score out of expected range: " << score;
          }
          score = 0.0;
        }
        vector<double> result;
        result.push_back(uid);
        result.push_back(tbeg * decodable_opts.frame_subsampling_factor);
        result.push_back(tend * decodable_opts.frame_subsampling_factor);
        result.push_back(score);
        result_writer.Write(key, result);
      }

      n_done++;
    }
    timing_stats.Print(online);

    KALDI_LOG << "Decoded " << num_done << " utterances, "
              << num_err << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
              << " per frame over " << num_frames << " frames.";
    delete decode_fst;
    delete word_syms; // will delete if non-NULL.
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
