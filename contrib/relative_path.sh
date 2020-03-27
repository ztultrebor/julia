#!/bin/sh
# This file is a part of Julia. License is MIT: https://julialang.org/license

# both $1 and $2 are absolute paths beginning with /
# returns relative path to $2/$target from $1/$source
# if the relative path to $2 would just be its basename, return `./$2`
# that is, the result always includes a directory component.

relpath () {
    [ $# -ge 1 ] && [ $# -le 2 ] || return 1
    current="${2:+"$1"}"
    target="${2:-"$1"}"
    [ "$target" != . ] || target=/
    target="/${target##/}"
    [ "$current" != . ] || current=/
    current="${current:="/"}"
    current="/${current##/}"
    appendix="${target##/}"
    relative=''
    while appendix="${target#"$current"/}"
        [ "$current" != '/' ] && [ "$appendix" = "$target" ]; do
        if [ "$current" = "$appendix" ]; then
            relative="${relative:-.}"
            echo "${relative#/}"
            return 0
        fi
        current="${current%/*}"
        relative="$relative${relative:+/}.."
    done
    relative="$relative${relative:+${appendix:+/}}${appendix#/}"
    [ "`basename "${relative}"`" != "${relative}" ] || relative="./${relative}"
    echo "${relative}"
}
relpath "$@"
