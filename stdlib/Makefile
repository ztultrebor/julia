default: install

SRCDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
JULIAHOME := $(abspath $(SRCDIR)/..)
SRCCACHE := $(abspath $(SRCDIR)/srccache)
BUILDDIR := .

include $(JULIAHOME)/Make.inc
include $(JULIAHOME)/deps/tools/common.mk
include $(JULIAHOME)/deps/tools/stdlib-external.mk
include $(JULIAHOME)/deps/tools/jll-rewrite.mk

VERSDIR := v$(shell cut -d. -f1-2 < $(JULIAHOME)/VERSION)

$(build_datarootdir)/julia/stdlib/$(VERSDIR):
	mkdir -p $@

# stdlibs that are vendored in-tree
STDLIBS = Base64 CRC32c Dates DelimitedFiles Distributed FileWatching \
          Future InteractiveUtils Libdl LibGit2 LinearAlgebra Logging \
          Markdown Mmap Printf Profile Random REPL Serialization SHA \
          SharedArrays Sockets SparseArrays SuiteSparse Test Unicode UUIDs

# stdlibs that are downloaded from their own repository
# Each has a $(name).version file denoting the exact version to use.
STDLIBS_EXT = Pkg Statistics

# We also include a number of JLLs here as standard libraries
JLL_EXT = MbedTLS_jll Zlib_jll LibCURL_jll LibSSH2_jll LibGit2_jll SuiteSparse_jll METIS_jll OpenBLAS_jll GMP_jll MPFR_jll
STDLIBS_EXT += $(JLL_EXT)

# stdlib-external generates the rules to download/extract tarballs of these external stdlibs
$(foreach module, $(STDLIBS_EXT), $(eval $(call stdlib-external,$(module),$(call uppercase,$(module)))))

# jll-rewrite generates the rules to rewrite JLL packages such that they
$(foreach module, $(JLL_EXT), $(eval $(call jll-rewrite,$(module),$(call uppercase,$(module)))))

# Generate symlinks to all stdlibs at usr/share/julia/stdlib/vX.Y/
$(foreach module, $(STDLIBS), $(eval $(call symlink_target,$$(JULIAHOME)/stdlib/$(module),$$(build_datarootdir)/julia/stdlib/$$(VERSDIR),$(module))))

STDLIBS_LINK_TARGETS := $(addprefix $(build_datarootdir)/julia/stdlib/$(VERSDIR)/,$(STDLIBS))

getall get: $(addprefix get-, $(STDLIBS_EXT))
install: $(addprefix install-, $(STDLIBS_EXT)) $(STDLIBS_LINK_TARGETS)
clean: $(addprefix clean-, $(STDLIBS_EXT)) $(CLEAN_TARGETS)
distclean: $(addprefix distclean-, $(STDLIBS_EXT)) clean

DEP_LIBS_STAGED := $(STDLIBS_EXT)
include $(JULIAHOME)/deps/tools/uninstallers.mk
