define artifact-install
# Target name is lowercased prefix, e.g. "MbedTLS_jll" -> "mbedtls"
$(2)_TARGET_NAME := $(or $(3),$(firstword $(subst _, ,$(call lowercase,$(1)))))

# If the Artifacts.toml file doesn't exist, we need to get it RIGHT NOW, before
# we can continue.  This is an unfortunate fact of how we've structured the Makefiles,
# separating `deps` from `stdlib` and not being able to express the dependency graph
# completely between the two branches of the source tree.
$(2)_ARTIFACTS_TOML := $(build_datarootdir)/julia/stdlib/$(VERSDIR)/$(1)/Artifacts.toml
$(devnull $(shell [ ! -e "$(build_datarootdir)/julia/stdlib/$(VERSDIR)/$(1)/Artifacts.toml" ] && $(MAKE) -C $(JULIAHOME)/stdlib install-$(1)))

$(2)_ARTIFACT_INFO := $$(shell $(PYTHON) $(JULIAHOME)/contrib/extract_artifact_info.py $$($(2)_ARTIFACTS_TOML) $(BB_TRIPLET_LIBGFORTRAN_CXXABI))
$(2)_TREEHASH      := $$(word 1,$$($(2)_ARTIFACT_INFO))
$(2)_URL           := $$(word 2,$$($(2)_ARTIFACT_INFO))
$(2)_TRIPLET       := $$(word 3,$$($(2)_ARTIFACT_INFO))
$(2)_ARTIFACT_DIR  := $(build_datarootdir)/julia/artifacts/$$($(2)_TREEHASH)

# How to download the artifacts tarball
$(SRCCACHE)/$(1)-$$($(2)_TREEHASH).tar.gz: $$($(2)_ARTIFACTS_TOML)
	$(JLDOWNLOAD) $$@ $$($(2)_URL)

# How to unpack the artifacts tarball
$$($(2)_ARTIFACT_DIR): $(SRCCACHE)/$(1)-$$($(2)_TREEHASH).tar.gz
	mkdir -p $$@
	$(JLCHECKSUM) $$<
	cd $$@ && $(TAR) -zxf $$<
	touch $$@

# Generate a manifest for this jll-unpacked artifact
UNINSTALL_$$($(2)_TARGET_NAME) := $(1) artifact-uninstaller $$($(2)_ARTIFACT_DIR)
$$(build_prefix)/manifest/$$($(2)_TARGET_NAME): $$($(2)_ARTIFACT_DIR) | $$(build_prefix)/manifest
	echo '$$(UNINSTALL_$$($(2)_TARGET_NAME))' > $$@

install-$$($(2)_TARGET_NAME): $$(build_prefix)/manifest/$$($(2)_TARGET_NAME)
.PHONY: install-$$($(2)_TARGET_NAME)

endef

define artifact-uninstaller
uninstall-$(strip $1):
	-rm -rf $(build_datarootdir)/julia/artifacts/$3
	-rm -f $$(build_prefix)/manifest/$(strip $1)
endef
