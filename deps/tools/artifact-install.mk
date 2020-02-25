# Define a set of targets for downloading an artifact from a JLL package.
#
# Parameters to this macro::
#   $1 = jll_name (e.g. OpenBLAS_jll)
#   $2 = target name (optional, e.g. "openblas")

define artifact-install
# Target name is lowercased prefix, e.g. "MbedTLS_jll" -> "mbedtls"
$(1)_TARGET_NAME := $(firstword $(subst _, ,$(call lowercase,$(1))))
# Convert from `MBEDTLS_JLL_SRC_DIR` to `MbedTLS_jll_SRC_DIR` for convenience
$(1)_SRC_DIR := $(BUILDDIR)/$$($(call uppercase,$(1)_SRC_DIR))

# If the Artifacts.toml file doesn't exist, we need to get it RIGHT NOW, before
# we can continue.  This is an unfortunate fact of how we've structured the Makefiles,
# separating `deps` from `stdlib` and not being able to express the dependency graph
# completely between the two branches of the source tree.
$(1)_ARTIFACTS_TOML := $$($(1)_SRC_DIR)/Artifacts.toml

# To get the artifacts tarball, we need to download and unpack the stdlib itself
$$($(1)_ARTIFACTS_TOML): | extract-$(1)

# We're going to need to extract information from the Artifacts.toml file pretty often;
# here's a bash snippet on how to do just that:
$(1)_GET_META_INFO := \
	artifact_info="$$$$($(PYTHON) $(JULIAHOME)/contrib/extract_artifact_info.py \
	                       $$($(1)_ARTIFACTS_TOML) $(BB_TRIPLET_LIBGFORTRAN_CXXABI))"; \
	treehash="$$$$(echo $$$${artifact_info} | cut -d' ' -f1)"; \
	url="$$$$(echo $$$${artifact_info} | cut -d' ' -f2)"; \
	artifact_dir="$(build_datarootdir)/julia/artifacts/$$$${treehash}"; \
	src_tarball="$(SRCCACHE)/$(1)-artifact-$$$${treehash}.tar.gz";

# How to download the artifacts tarball
$$($(1)_SRC_DIR)/artifact-downloaded: $$($(1)_ARTIFACTS_TOML)
	@$$($(1)_GET_META_INFO) \
	$(JLDOWNLOAD) "$$$${src_tarball}" "$$$${url}"
	touch "$$@"

# Generate a manifest for this jll-unpacked artifact, calling it just `openblas`
# but having it depend on `extract-artifact-OpenBLAS_jll`.
UNINSTALL_$$($(1)_TARGET_NAME) := $(1) artifact-uninstaller $$($(1)_ARTIFACT_DIR)
$$(build_prefix)/manifest/$$($(1)_TARGET_NAME): $$($(1)_SRC_DIR)/artifact-downloaded | $$(build_prefix)/manifest
	@$$($(1)_GET_META_INFO) \
	mkdir -p $$$${artifact_dir}; \
	$(JLCHECKSUM) "$$$${src_tarball}" ; \
	cd "$$$${artifact_dir}"; \
	$(TAR) -zxf "$$$${src_tarball}"
	@echo '$$(UNINSTALL_$$($(1)_TARGET_NAME))' > "$$@"

get-artifact-$(1): $$($(1)_SRC_DIR)/artifact-downloaded

# `install-openblas` relies on `manifest/openblas`, which relies on `extract-artifact-OpenBLAS_jll`
install-$$($(1)_TARGET_NAME): $$(build_prefix)/manifest/$$($(1)_TARGET_NAME)
.PHONY: install-$$($(1)_TARGET_NAME)

# Define libdir for this target, for people who want to link against us:
# To match conventions, we define e.g. `OPENBLAS_LIBDIR`.  Note that this rule
# is explicitly NOT expanded right now (e.g. it uses `=` instead of `:=`).
# This is because the `Artifacts.toml` file may not exist at this point;
# we need to expand this value only _after_ the buildsystem has bothered
# to download those files for us.  This means that every time someone tries
# to use (e.g. `OPENBLAS_LIBDIR`), it will invoke a `python` process to parse
# the toml file.  This isn't ideal, but it's not a dealbreaker either.
$$(eval $$(call uppercase,$$($(1)_TARGET_NAME))_LIBDIR = $$$$(shell $$$$($(1)_GET_META_INFO) echo $$$$$$$${artifact_dir}))
endef

# Uninstaller that determines the artifact directory of a JLL, then deletes it
define artifact-uninstaller
uninstall-$(strip $1):
	@if [ -e "$$($(1)_ARTIFACTS_TOML)" ]; then \
		$$($(1)_GET_META_INFO) \
		-rm -rf "$$$${artifact_dir}" ;\
	fi
	-rm -f $$(build_prefix)/manifest/$(strip $1)
endef
