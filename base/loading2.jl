using Base: PkgId, UUID, SHA1
using Base: load_path, dummy_uuid, manifest_names, isfile_casesensitive
using Base: entry_point_and_project_file_inside, project_file_name_uuid, env_project_file, entry_path
using Base: re_section,
    #re_array_of_tables,
    re_section_deps,
    re_section_capture,
    re_subsection_deps,
    re_key_to_string,
    re_uuid_to_string,
    re_name_to_string,
    re_path_to_string,
    re_hash_to_string,
    re_manifest_to_string,
    re_deps_to_any

const PkgMap = Dict{Symbol, PkgId}
const PkgInfo = Tuple{PkgMap, Dict{PkgId, PkgMap}, Dict{PkgId, Any #=path::String or hash::SHA1=#}}

# parse a Manifest file
function explicit_manifest_deps_get(manifest_file::String)::Tuple{Dict{PkgId, PkgMap}, Dict{PkgId, Any}}
    info = Dict{PkgId, PkgMap}()
    paths = Dict{PkgId, Any}()
    open(manifest_file) do io
        id_map = PkgMap()
        later_id = Vector{PkgId}()
        later_deps = Vector{Vector{Any}}()
        uuid = deps = path = hash = nothing
        name = nothing
        state = :top
        function finish_add()
            name === nothing && return
            uuid === nothing && @warn "Missing UUID for $name may not be handled correctly" # roots should be the toplevel list
            id = PkgId(uuid, name)
            if deps isa PkgMap
                info[id] = deps
            elseif deps !== nothing
                deps_list = nothing
                # deps is a String of names, later we need to convert this to a Dict of UUID
                # TODO: handle inline table syntax
                if deps[1] == '[' || deps[end] == ']'
                    deps_list = Meta.parse(deps, raise=false)
                    if Meta.isexpr(deps_list, :vect) && all(x -> isa(x, String), deps_list.args)
                        deps_list = deps_list.args
                    end
                end
                if deps_list === nothing
                    @warn "Unexpected TOML deps format:\n$deps"
                else
                    push!(later_id, id)
                    push!(later_deps, deps_list)
                end
            end
            id_map[Symbol(name)] = id
            if path !== nothing
                path = normpath(abspath(dirname(manifest_file), path))
                path = entry_path(path, name)
                paths[id] = path
            elseif hash !== nothing
                paths[id] = hash
            end
            nothing
        end
        # parse the manifest file, looking for uuid, name, and deps sections
        for line in eachline(io)
            if (m = match(re_section_capture, line)) !== nothing
                finish_add()
                uuid = deps = path = hash = nothing
                name = String(m.captures[1])
                state = :stanza
            elseif state == :stanza
                if (m = match(re_uuid_to_string, line)) !== nothing
                    uuid = UUID(m.captures[1])
                elseif (m = match(re_path_to_string, line)) != nothing
                    path = String(m.captures[1])
                elseif (m = match(re_hash_to_string, line)) != nothing
                    hash = SHA1(m.captures[1])
                elseif (m = match(re_deps_to_any, line)) !== nothing
                    deps = String(m.captures[1])
                elseif occursin(re_subsection_deps, line)
                    state = :deps
                    deps = PkgMap()
                elseif occursin(re_section, line)
                    state = :other
                end
            elseif state == :deps
                if (m = match(re_key_to_string, line)) !== nothing
                    d_name = String(m.captures[1])
                    d_uuid = UUID(m.captures[2])
                    deps[Symbol(d_name)] = PkgId(d_uuid, d_name)
                end
            end
        end
        finish_add()
        # now that we have a map of name => uuid,
        # build the rest of the map
        for i = 1:length(later_id)
            id = later_id[i]
            deps = later_deps[i]
            deps_map = PkgMap()
            for dep in deps
                dep = Symbol(dep::String)
                deps_map[dep] = id_map[dep]
            end
            info[id] = deps_map
        end
        nothing
    end
    return info, paths
end

# parse a Project file
# return the set of entry point that exist in an explicit environment
function explicit_project_deps_get(project_file::String)::PkgInfo
    roots = PkgMap()
    info = Dict{PkgId, PkgMap}()
    paths = Dict{PkgId, Any}()
    open(project_file) do io
        dir = abspath(dirname(project_file))
        root_name = nothing
        manifest_file = nothing
        root_uuid = dummy_uuid(project_file)
        state = :top
        for line in eachline(io)
            if occursin(re_section, line)
                state = occursin(re_section_deps, line) ? :deps : :other
            elseif state == :top
                if (m = match(re_name_to_string, line)) != nothing
                    root_name = String(m.captures[1])
                elseif (m = match(re_uuid_to_string, line)) != nothing
                    root_uuid = UUID(m.captures[1])
                elseif (m = match(re_manifest_to_string, line)) != nothing
                    manifest_file = normpath(joinpath(dir, m.captures[1]))
                end
            elseif state == :deps
                if (m = match(re_key_to_string, line)) != nothing
                    name = String(m.captures[1])
                    uuid = UUID(m.captures[2])
                    roots[Symbol(name)] = PkgId(uuid, name)
                end
            end
        end
        root_name !== nothing && (roots[Symbol(root_name)] = PkgId(root_uuid, root_name))

        if manifest_file === nothing
            for mfst in manifest_names
                manifest_file = joinpath(dir, mfst)
                if isfile_casesensitive(manifest_file)
                    info, paths = explicit_manifest_deps_get(manifest_file)
                    break
                end
            end
        elseif isfile_casesensitive(manifest_file)
            info, paths = explicit_manifest_deps_get(manifest_file)
        end
        nothing
    end
    return roots, info, paths
end


# return the set of entry point that exist in an implicit environment
function implicit_project_deps_get(dir::String)::PkgInfo
    roots = PkgMap()
    info = Dict{PkgId, PkgMap}()
    paths = Dict{PkgId, Any}()
    function add_path(name, path, project_file::Nothing)
        id = PkgId(name)
        info[id] = roots
        roots[Symbol(name)] = id
        paths[id] = path
        nothing
    end
    function add_path(name, path, project_file)
        id = project_file_name_uuid(project_file, name)
        if id.name == name
            deps = explicit_project_deps_get(project_file)[1] # ignore a Manifest inside a package
            info[id] = deps
            roots[Symbol(name)] = id
            paths[id] = path
        end
        nothing
    end
    for fname in readdir(dir)
        fpath = joinpath(dir, fname)
        s = stat(fpath)
        isjl = endswith(fname, ".jl")
        name = isjl ? fname[1:prevind(fname, end - 2)] : fname
        if isjl && isfile(s)
            add_path(name, fpath, nothing)
        elseif isdir(s)
            if isjl
                path, project_file = @show (entry_point_and_project_file_inside(fpath, name)..., name)
                path === nothing || add_path(name, path, project_file)
            end
            path, project_file = @show (entry_point_and_project_file_inside(fpath, fname)..., name)
            path === nothing || add_path(fname, path, project_file)
        end
    end
    return roots, info, paths
end

# return the set of entry point that exist in an environment
function project_deps_get(env::String)::PkgInfo
    project_file = env_project_file(env)
    if project_file isa String
        # use project and manifest files
        return explicit_project_deps_get(project_file)
    elseif project_file
        # if env names a directory, search it
        return implicit_project_deps_get(env)
    end
    return (PkgMap(), Dict{PkgId, PkgMap}(), Dict{PkgId, Any}())
end

# identify_package computes the PkgId for `name` from toplevel context
# by looking through the Project.toml files and directories
function identify_package()::Vector{PkgInfo}
    return PkgInfo[ project_deps_get(env) for env in load_path() ]
end

function identify_package(envs::Vector{PkgInfo}, name::String)::Union{Nothing,PkgId}
    name = Symbol(name)
    for env in envs
        uuid = get(envs[1], name, nothing)
        uuid === nothing || return uuid # found--return it
    end
    return nothing
end

function identify_package(envs::Vector{PkgInfo}, where::PkgId, name::String)::Union{Nothing,PkgId}
    name = Symbol(name)
    for env in envs
        if where in keys(envs[3]) # found where--look at deps and return name
            deps = get(envs[2], where, nothing)
            deps === nothing && return nothing # no deps
            uuid = get(deps, name, nothing)
            return uuid
        end
    end
    return nothing
end

function locate_package(envs::Vector{PkgInfo}, pkg::PkgId)::Union{Nothing,String}
    for env in envs
        if pkg.uuid === nothing
            # see if we can find a `uuid` for pkg, and then restart the search of the manifests if so
            uuid = get(env[1], Symbol(pkg.name), nothing)
            uuid !== nothing && uuid.uuid !== nothing && return locate_package(envs, uuid)
        end
        path = get(envs[3], pkg, nothing)
        path === nothing || return path
    end
    return nothing
end

function find_package(envs::Vector{PkgInfo}, args...)
    pkg = identify_package(envs, args...)
    pkg === nothing && return nothing
    return locate_package(envs, pkg)
end
