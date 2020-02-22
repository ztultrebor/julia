#$(call bb-install, \
#    1 target, \               # name (lowercase)
#    2 gfortran, \             # signifies a GCC ABI (e.g. libgfortran version) dependency
#    3 cxx11)                  # signifies a cxx11 ABI dependency

define bb-install
TRIPLET_VAR := BB_TRIPLET
UPPER_VAR := $(call uppercase,$(1))
ifeq ($(2),true)
TRIPLET_VAR := $$(TRIPLET_VAR)_LIBGFORTRAN
endif
ifeq ($(3),true)
TRIPLET_VAR := $$(TRIPLET_VAR)_CXXABI
endif
$$(UPPER_VAR)_BB_TRIPLET := $$($$(TRIPLET_VAR))
$$(UPPER_VAR)_BB_URL := $$($$(UPPER_VAR)_BB_URL_BASE)/$$($$(UPPER_VAR)_BB_NAME).$$($$(UPPER_VAR)_BB_TRIPLET).tar.gz
$$(UPPER_VAR)_BB_BASENAME := $$($$(UPPER_VAR)_BB_NAME)-$$($$(UPPER_VAR)_BB_REL).$$($$(UPPER_VAR)_BB_TRIPLET).tar.gz

$$(BUILDDIR)/$$($$(UPPER_VAR)_BB_NAME):
	mkdir -p $$@

$$(SRCCACHE)/$$($$(UPPER_VAR)_BB_BASENAME): | $$(SRCCACHE)
	echo $$($$(UPPER_VAR)_BB_URL)
	$$(JLDOWNLOAD) $$@ $$($$(UPPER_VAR)_BB_URL)

stage-$(strip $1): $$(SRCCACHE)/$$($$(UPPER_VAR)_BB_BASENAME)
install-$(strip $1): $$(build_prefix)/manifest/$(strip $1)

reinstall-$(strip $1):
	+$$(MAKE) uninstall-$(strip $1)
	+$$(MAKE) stage-$(strip $1)
	+$$(MAKE) install-$(strip $1)

UNINSTALL_$(strip $1) := $$($$(UPPER_VAR)_BB_BASENAME:.tar.gz=) bb-uninstaller

$$(build_prefix)/manifest/$(strip $1): $$(SRCCACHE)/$$($$(UPPER_VAR)_BB_BASENAME) | $(build_prefix)/manifest
	-+[ ! -e $$@ ] || $$(MAKE) uninstall-$(strip $1)
	$$(JLCHECKSUM) $$<
	mkdir -p $$(build_prefix)
	$(UNTAR) $$< -C $$(build_prefix)
	echo '$$(UNINSTALL_$(strip $1))' > $$@

clean-bb-download-$(1):
	rm -f $$(SRCCACHE)/$$($$(UPPER_VAR)_BB_BASENAME)

clean-$(1):
distclean-$(1): clean-bb-download-$(1)
get-$(1): $$(SRCCACHE)/$$($$(UPPER_VAR)_BB_BASENAME)
extract-$(1):
configure-$(1):
compile-$(1): get-$(1)
fastcheck-$(1):
check-$(1):

.PHONY: clean-bb-$(1)

endef

define bb-uninstaller
uninstall-$(strip $1):
	-cd $$(build_prefix) && rm -fdv -- $$$$($$(TAR) -tzf $$(SRCCACHE)/$2.tar.gz --exclude './$$$$')
	-rm $$(build_prefix)/manifest/$(strip $1)
endef
