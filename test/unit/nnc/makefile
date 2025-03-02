include ../../../lib/config.mk

export LSAN_OPTIONS=suppressions=known-leaks.txt
LDFLAGS := -L"../../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../../lib" -I"../../" $(CFLAGS)
TARGETS = tfb.tests tensor.tests forward.tests backward.tests gradient.tests graph.tests winograd.tests transform.tests symbolic.graph.tests autograd.tests autograd.vector.tests while.tests tape.tests while.backward.tests case_of.tests case_of.backward.tests numa.tests tensor.bind.tests broadcast.tests reduce.tests batch.norm.tests layer.norm.tests dropout.tests crossentropy.tests dynamic.graph.tests simplify.tests symbolic.graph.compile.tests rand.tests graph.io.tests cnnp.core.tests minimize.tests custom.tests parallel.tests dataframe.tests dataframe.addons.tests compression.tests gemm.tests index.tests swish.tests upsample.tests smooth_l1.tests roi_align.tests nms.tests compare.tests concat.tests cblas.tests micro.tests loss.tests histogram.tests group.norm.tests gelu.tests attention.tests palettize.tests rmsnorm.tests complex.tests pad.tests

ifeq ($(shell uname 2>/dev/null || echo Unknown),Darwin)
	LDFLAGS += -Wl,-U,___test_case_setup,-U,___test_case_teardown
endif

TARGET_SRCS := $(patsubst %,%.c,$(TARGETS))

.PRECIOUS: $(patsubst %,%.o,$(TARGETS))

.PHONY: release all test clean dep

release: all

include ../../../lib/scheme.mk

all: $(TARGETS)

test:
	@for test in $(TARGETS) ; do LLVM_PROFILE_FILE="$$test.profraw" ./"$$test" || exit ; done

clean:
	${MAKE} clean -C ../../../lib ; rm -f *.o $(TARGETS)

%.tests: %.tests.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../../lib

%.o: %.c
	$(CC) $< -o $@ -c $(CFLAGS)

.gitignore:
	echo $(TARGETS) | tr ' ' '\n' > .gitignore

dep: .dep.mk
.dep.mk: $(TARGET_SRCS)
	$(CC) $(CFLAGS) -MM $^ > .dep.mk

-include .dep.mk
