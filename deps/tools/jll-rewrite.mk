# Define a set of targets for rewriting a JLL package to not depend on `Pkg`;
# or anything outside of `Base`, really.
#
# Parameters to the stdlib-external macro:
#
#   $1 = stdlib_name
#   $2 = var_prefix (by convention, use upper cased stdlib_name)

define jll-rewrite

# We need to eliminate the JLL package's dependency on `Pkg`; this is because we need to
# load things like `LibGit2_jll`, needed by `LibGit2`, which is itself needed by `Pkg`.
# JLL packages only use Pkg for two things; introspecting the current platform so that
# it can choose which wrapper to load, and downloading artifacts if they're missing.
# We know all the necessary information beforehand, so we modify JLL packages a bit here
# to reduce the amount of work they're doing.  We choose, at compile-time, the platform
# to load, then we `sed` out the `artifact""` call and substitute an equivalent path
# based off of `Sys.STDLIB`.  Who doesn't love some horrific Make/bash/sed spaghetti?
#
# - ARTIFACT_INFO: extracted information from Artifacts.toml; essentially working around
#   the fact that we don't have a Julia `Pkg` around to parse it for us.
# - TRIPLET: the triplet selected from the available options in the JLL
# - WRAPPER: the chosen wrapper `.jl` file within `src/wrappers`, which will get copied
#   to GEN_SRC and used as the overall JLL module.
# - REL_PATH: the julia code that points to the known-good artifact directory
$$(BUILDDIR)/$$($2_SRC_DIR)/jll-rewritten: $$(BUILDDIR)/$$($2_SRC_DIR)/Artifacts.toml | extract-$1
	-GEN_SRC="$$(BUILDDIR)/$$($2_SRC_DIR)/src/$(1).jl"; \
	ARTIFACT_INFO="$$$$($(PYTHON) $(JULIAHOME)/contrib/extract_artifact_info.py "$$<" $(BB_TRIPLET_LIBGFORTRAN_CXXABI))"; \
	TREEHASH="$$$$(echo $$$${ARTIFACT_INFO} | cut -d ' ' -f1)"; \
	TRIPLET="$$$$(echo $$$${ARTIFACT_INFO} | cut -d ' ' -f3)"; \
	WRAPPER="$$(BUILDDIR)/$$($2_SRC_DIR)/src/wrappers/$$$${TRIPLET}.jl"; \
	REL_PATH="joinpath(dirname(dirname(Sys.STDLIB)), \\\"artifacts\\\", \\\"$$$$TREEHASH\\\")"; \
	echo "module $(1)" > "$$$${GEN_SRC}"; \
	echo "using Libdl" >> "$$$${GEN_SRC}"; \
	echo "const PATH_list = String[]; const LIBPATH_list = String[];" >> "$$$${GEN_SRC}"; \
	sed -e "s/artifact\\\"$(subst _jll,,$(1))\\\"/$$$${REL_PATH}/" <"$$$${WRAPPER}" >>"$$$${GEN_SRC}"; \
	echo "end" >> "$$$${GEN_SRC}"
	touch $$@

# Add rewrite rule to list of things necessary to satisfy `install-$1`
install-$1: $$(BUILDDIR)/$$($2_SRC_DIR)/jll-rewritten
endef
