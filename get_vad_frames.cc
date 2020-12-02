#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Compute vaded frames numbers.\n"
        "Usage: get_vad_frames [options...] <feat_rspecifier>"
        "<vad_rspecifier> <utt2dur_wspecifier>\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feat_rspecifier = po.GetArg(1);
    std::string vad_rspecifier = po.GetArg(2);
    std::string utt2dur_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    RandomAccessBaseFloatVectorReader vad_reader(vad_rspecifier);
    DoubleWriter utt2dur_writer(utt2dur_wspecifier);

    int32 num_done = 0;

    for (;!feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();

      const Vector<BaseFloat> &voiced = vad_reader.Value(utt);

      int32 dim = 0;
      for (int32 i = 0; i < voiced.Dim(); i++)
        if (voiced(i) != 0.0)
          dim++;    

      utt2dur_writer.Write(utt, dim);
      num_done++;
    } 
  } catch(...) {
    std::cout << "Bad line" << std::endl;
    return -1;
    }
}
