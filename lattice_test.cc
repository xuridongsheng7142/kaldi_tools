#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

#include <fstream>//ifstream读文件，ofstream写文件，fstream读写文件
#include <map>


std::vector<string> split(const string& str, const string& pattern)
{
        std::vector<string> ret;
        if(pattern.empty()) return ret;
        size_t start=0,index=str.find_first_of(pattern,0);
        while(index!=str.npos)
        {
            if(start!=index)
                ret.push_back(str.substr(start,index-start));
            start=index+1;
            index=str.find_first_of(pattern,start);
        }
        if(!str.substr(start).empty())
            ret.push_back(str.substr(start));
        return ret;
}

namespace kaldi {

void CompactLattice_fix(CompactLattice *clat, 
                        std::string word_syms_filename) {

  using namespace fst;

  typedef CompactLatticeArc Arc;

  std::ifstream in(word_syms_filename);
  std::string line;
  std::vector<string> info;
  std::string word;
  std::string scale;

  std::map<string, string> namemap;  // <word_id, scale> : 91487 8

  while (getline (in, line)) {
      info = split(line, " ");
      if (info.size() != 1 && info.size() != 2) {
          std::cout << "word_syms error, please check!!!" << std::endl;
          exit(1);
      }
      word = info[0];
      if (info.size() == 1) {
          scale = "1";
      }
      else {
          scale = info[1];
      }
      namemap[word] = scale;
  }

  for (int32 state = 0; state < clat->NumStates(); state++) {
    for (MutableArcIterator<CompactLattice> aiter(clat, state);
         !aiter.Done(); aiter.Next()) {
        Arc arc(aiter.Value());
        if (arc.ilabel != 0) { // if there is a word on this arc
            if(namemap.find(std::to_string(arc.ilabel)) != namemap.end()){
//                int word_index = arc.ilabel;
                int scale = atoi(namemap[std::to_string(arc.ilabel)].c_str());
                LatticeWeight weight = arc.weight.Weight();
                weight.SetValue1(weight.Value1() - (float) (scale * 10));
                arc.weight.SetWeight(weight);
//                std::cout << word_index << " " << scale << std::endl;
            }
        }
        aiter.SetValue(arc);
    }
  }
}

};

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Generate 1-best path through lattices; output as transcriptions and alignments\n"
        "Note: if you want output as FSTs, use lattice-1best; if you want output\n"
        "with acoustic and LM scores, use lattice-1best | nbest-to-linear\n"
        "Usage: lattice-best-path [options]  <lattice-rspecifier> [ <transcriptions-wspecifier> [ <alignments-wspecifier>] ]\n"
        " e.g.: lattice-best-path --acoustic-scale=0.1 ark:1.lats 'ark,t:|int2sym.pl -f 2- words.txt > text' ark:1.ali\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() < 1 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        word_syms_filename = po.GetOptArg(2);

    std::string lats_wspecifier = "ark,t:1.t";

    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      CompactLattice clat = clat_reader.Value();
      clat_reader.FreeCurrent();
      CompactLattice_fix(&clat, word_syms_filename);
      compact_lattice_writer.Write(key, clat);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
