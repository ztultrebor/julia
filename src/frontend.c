// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "julia.h"
#include "julia_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Parsing

// Pointer to the current parser
jl_parse_func_t jl_current_parser = NULL;

JL_DLLEXPORT void jl_set_parser(jl_parse_func_t parser)
{
    jl_current_parser = parser;
}

JL_DLLEXPORT jl_value_t *jl_parse(jl_value_t *text, jl_value_t *filename,
                                  int start_pos, int rule)
{
    return (*jl_current_parser)(text, filename, start_pos, rule);
}

// C API
// parse an entire string like a file, reading multiple expressions
JL_DLLEXPORT jl_value_t *jl_parse_all(const char *str, size_t len,
                                      const char *filename, size_t filename_len)
{
    jl_value_t *text = NULL;
    jl_value_t *filename_ = NULL;
    JL_GC_PUSH2(&text, &filename_);
    text = jl_pchar_to_string(str, len);
    filename_ = jl_pchar_to_string(filename, filename_len);
    jl_value_t *p = jl_parse(text, filename_, 1, JL_PARSE_TOPLEVEL);
    JL_GC_POP();
    return jl_svecref(p, 0);
}

// this is for parsing one expression out of a string, keeping track of
// the current position.
JL_DLLEXPORT jl_value_t *jl_parse_string(const char *str, size_t len,
                                         int pos0, int greedy)
{
    jl_value_t *text = NULL;
    jl_value_t *filename_ = NULL;
    JL_GC_PUSH2(&text, &filename_);
    text = jl_pchar_to_string(str, len);
    filename_ = jl_cstr_to_string("none");
    jl_value_t *result = jl_parse(text, filename_, pos0+1,
                                  greedy ? JL_PARSE_STATEMENT : JL_PARSE_ATOM);
    JL_GC_POP();
    return result;
}

// deprecated
JL_DLLEXPORT jl_value_t *jl_parse_input_line(const char *str, size_t len,
                                             const char *filename, size_t filename_len)
{
    return jl_parse_all(str, len, filename, filename_len);
}

#ifdef __cplusplus
}
#endif
