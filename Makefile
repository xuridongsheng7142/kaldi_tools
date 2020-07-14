
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = get_vad_frames gen_decode nnet3-compute test wav2vad compute-mfcc-pitch_feats wav2gen

OBJFILES =

TESTFILES =

cuda-compiled.o: ../kaldi.mk

ADDLIBS = ../nnet3/kaldi-nnet3.a ../hmm/kaldi-hmm.a ../feat/kaldi-feat.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a ../ivector/kaldi-ivector.a  \
          ../chain/kaldi-chain.a \
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a

include ../makefiles/default_rules.mk
