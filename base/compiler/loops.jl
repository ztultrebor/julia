enterif(ex, head, argidx) = isexpr(ex, head) && length(ex.args) >= argidx ?
    ex.args[argidx] : ex

isglobalref(g, mod, name) = isa(g, GlobalRef) && g.mod === mod && g.name === name

getlhs(src, id) = isexpr(src.code[id], :(=)) ? src.code[id].args[1] : SSAValue(id)
getrhs(src, id) = enterif(src.code[id], :(=), 2)

# Extract loop information from a call to `iterate`
#    iterator, iteratortype, itervar, bodystart, bodystop, loopexit = extract_iterate(src, id)
# where
#   - iterator::Union{Slot,SSAValue} is the iterator
#   - iteratortype is the type of the iterator
#   - itervar::Union{Slot,SSAValue} is the `i` in `for i = ...`
#   - bodystart::Int is the id of the first statement of the loop body
#   - bodystop::Int is the id of the last statement of the loop body
#   - loopexit::Int is the id of the first post-loop statement
function extract_iterate(src, id)
    # lhs = getlhs(src, id)
    rhs = getrhs(src, id)
    @assert isexpr(rhs, :call) && length(rhs.args) == 2
    @assert isglobalref(rhs.args[1], Main.Base, :iterate)
    iterator = rhs.args[2]
    iteratortype = if isa(iterator, SSAValue)
        src.ssavaluetypes[iterator.id]
    elseif isa(iterator, SlotNumber) || isa(iterator, TypedSlot)
        src.slottypes[iterator.id]
    else
        Any
    end
    # Identify the loop terminus by looking for the :goto
    idgoto = id
    while !isexpr(src.code[idgoto+=1], :gotoifnot) end
    loopexit = src.code[idgoto].args[2]
    # Extract the iteration variable and identify the loop body start
    bodystart = idgoto + 1
    if isa(src.code[bodystart], TypedSlot)
        bodystart += 1
    end
    rhs = getrhs(src, bodystart)
    @assert isexpr(rhs, :call) && length(rhs.args) == 3 &&
        isglobalref(rhs.args[1], Core, :getfield) &&
        rhs.args[3] == 1  # tuple-destructuring (iterator item)
    itervar = getlhs(src, bodystart)
    bodystart += 1
    rhs = getrhs(src, bodystart)
    @assert isexpr(rhs, :call) && length(rhs.args) == 3 &&
        isglobalref(rhs.args[1], Core, :getfield) &&
        rhs.args[3] == 2  # tuple-destructuring (iterator state)
    bodystart += 1
    # Identify the inner call to iterate
    bodystop = loopexit - 1
    rhs = getrhs(src, bodystop)
    while bodystop > 1 && !(isexpr(rhs, :call) && length(rhs.args) == 3) || !isglobalref(rhs.args[1], Main.Base, :iterate)
        bodystop -= 1
        rhs = getrhs(src, bodystop)
    end
    return iterator, iteratortype, itervar, bodystart, bodystop-1, loopexit
end

# For manipulating loops (and especially nested loops), it makes sense
# to transiently go back to a nested block structure to avoid lots of
# `insert!` and `deleteat!` shuffling.

# Hold the parts of a CodeInfo that we may rewrite
struct StmtBlock
    stmts::Vector{Any}
    codelocs::Vector{Int32}
    codelocdefault::Int32       # so empty blocks can be expanded correctly
    ssavaluetypes::Vector{Any}
    # Note: no ssaflags
    ssavalueids::Vector{Int32}  # the ids that `stmts` were created with
end
StmtBlock(codelocdefault::Integer) =
    StmtBlock([], Int32[], Int32(codelocdefault), [], Int32[])
StmtBlock(src::CodeInfo, rng::UnitRange) =
    StmtBlock([copy_exprs(src.code[i]) for i in rng],
              src.codelocs[rng],
              src.codelocs[first(rng)],
              src.ssavaluetypes[rng],
              Int32[i for i in rng])

function addstmt!(sb::StmtBlock, @nospecialize(stmt), @nospecialize(typ), id::Integer)
    push!(sb.stmts, stmt)
    push!(sb.codelocs, isempty(sb.codelocs) ? sb.codelocdefault : sb.codelocs[end])
    push!(sb.ssavaluetypes, typ)
    push!(sb.ssavalueids, id)
    return id+1
end

function append!(dest::StmtBlock, src::StmtBlock)
    append!(dest.stmts, src.stmts)
    append!(dest.codelocs, src.codelocs)
    append!(dest.ssavaluetypes, src.ssavaluetypes)
    append!(dest.ssavalueids, src.ssavalueids)
    return dest
end


mutable struct PrePostBlock
    pre::StmtBlock
    body::Union{Nothing,StmtBlock,PrePostBlock}
    post::StmtBlock
end

function append!(dest::StmtBlock, src::PrePostBlock)
    append!(dest, src.pre)
    append!(dest, src.body)
    append!(dest, src.post)
    return dest
end


#     block, enterid, item, itemtype, state, statetype, enterid, nextid = iterate_block(...)
# Create a block of statements implementing an `iterate` call.
# Call it once for the loop entrance and once for the exit; for the
#   exit you need to pass `state` and `statetype`.
# Arguments:
#   - iter is the reference to the iterator
#   - itertype is typeof(dereferenced iter)
#   - slotid is the slot number to which to assign the return value of
#     `iterate`, or an Int=>Symbol pair also specifying the name
#   - startloc is the starting codeloc
#   - nextid is the starting statement #/SSAValue
#   - state is a reference to the iterator state, or `nothing` on entrance
#   - statetype is typeof(dereferenced state), or `nothing` on entrance
# and
#  - block is the created StmtBlock
#  - enterid is the SSAValue of the entrance to the loop body, or `nothing` on exit
#  - item::SSAValue is a reference to the item returned from `iterate`
#  - itemtype::Type typeof(dererenced item)
#  - state::SSAValue is a reference to the state returned from `iterate`
#  - statetype::Type typeof(dereferenced state)
#  - nextid::Int is the next free statement #/SSAValue
function iterate_block!(src::CodeInfo, iter::Union{Slot,SSAValue}, itertype::Type, slotinfo::Union{Int,Pair{Int,Symbol}}, startloc::Int, enterid::Int, exitid::Int, nextid::Int, state::Union{Nothing,SSAValue}=nothing, statetype::Union{Nothing,Type}=nothing)
    if isa(slotinfo, Int)
        slotid, slotname = slotinfo, Symbol("")
    else
        slotid, slotname = slotinfo.first, slotinfo.second
    end
    block = StmtBlock(startloc)
    # Add the `iterate(iter)` or `iterate(iter, state)` call
    iterret = SlotNumber(slotid)
    if state === nothing
        iterrettype = return_type(Main.Base.iterate, Tuple{itertype})
        nextid = addstmt!(block,
                          Expr(:(=), iterret, Expr(:call, GlobalRef(Main.Base, :iterate), iter)),
                          iterrettype,
                          nextid)
    else
        iterrettype = return_type(Main.Base.iterate, Tuple{itertype,statetype})
        nextid = addstmt!(block,
                          Expr(:(=), iterret, Expr(:call, GlobalRef(Main.Base, :iterate), iter, state)),
                          iterrettype,
                          nextid)
    end
    if !isassigned(src.slotnames, slotid)
        @assert length(src.slotnames) == slotid - 1
        # push!(src.slotnames, Symbol(Main.Base.string(slotname, "[", i, "]")))
        push!(src.slotnames, slotname)
        push!(src.slotflags, 0x02)  # 0x02 = assigned
        push!(src.slottypes, iterrettype)
    end
    # %d === nothing
    addstmt!(block,  # deliberately don't increment nextid
             Expr(:call, GlobalRef(Core, :(===)), iterret, nothing),
             Bool,
             nextid)
    # Base.not_int(%d)
    addstmt!(block,
             Expr(:call, GlobalRef(Main.Base, :not_int), SSAValue(nextid)),
             Bool,
             nextid+1)
    # goto #x if not %y
    nextid = addstmt!(block,
                      Expr(:gotoifnot, SSAValue(nextid+1), exitid),
                      Any,
                      nextid+2)
    if isa(iterrettype, Union)
        itemstatetype = iterrettype.a === Nothing ? iterrettype.b : iterrettype.a
        itemtype  = itemstatetype.parameters[1]
        statetype = itemstatetype.parameters[2]
    else
        itemstatetype = itemtype = statetype = Any
    end
    item = nothing
    if state === nothing
        # item, state = iterret
        enterid = nextid
        iterretssa = SSAValue(nextid)
        # Union-splitting via insertion of a TypedSlot
        nextid = addstmt!(block,
                          TypedSlot(iterret.id, itemstatetype),
                          itemstatetype,
                          nextid)
        item = SSAValue(nextid)
        nextid = addstmt!(block,
                          Expr(:call, GlobalRef(Core, :getfield), iterretssa, 1),
                          itemtype,
                          nextid)
        state = SSAValue(nextid)
        nextid = addstmt!(block,
                          Expr(:call, GlobalRef(Core, :getfield), iterretssa, 2),
                          statetype,
                          nextid)
    else
        # Jump back to top of loop
        nextid = addstmt!(block,
                          GotoNode(enterid),
                          Any,
                          nextid)
        enterid = nothing
    end
    return block, enterid, item, itemtype, state, statetype, nextid
end

# Convert an iteration over a CartesianIndices into a set of nested loops
# `src.code[id]` must contain the call to `iterate(::CartesianIndices)`
function expand_cartesian_loop!(src::CodeInfo, id::Int)
    @assert isempty(src.ssaflags)
    @assert !isempty(src.ssavaluetypes)  # puzzlingly, src.inferred might still be false
    println(src)
    iterator, iteratortype, itervar, bodystart, bodystop, loopexit = extract_iterate(src, id)
    @assert iteratortype <: Main.Base.CartesianIndices
    # slotname = isa(iterator, Slot) ? Main.Base.string("#", Main.Base.String(src.slotnames[iterator.id])) : Main.Base.String(gensym())
    N = iteratortype.parameters[1]           # number of loops (dimensionality)
    tt = iteratortype.parameters[2]
    # Extract the code chunks
    wholepre  = StmtBlock(src, 1:id-1)
    body = StmtBlock(src, bodystart:bodystop)
    wholepost = StmtBlock(src, loopexit:length(src.code))
    nextid = length(src.code) + 1  # the next-to-be-created SSAValue.id
    # Extract the indices into separate SSAValues
    destruct, indicesid = StmtBlock(src.codelocs[id]), nextid
    ## Get the .indices field of the CartesianIterator
    nextid = addstmt!(destruct,
                      Expr(:call, GlobalRef(Main.Base, :getproperty), iterator, :indices),
                      tt,
                      nextid)
    ## Get the elements of the indices tuple
    iteratorids = Array{SSAValue}(undef, N)  # id of each 1d iterator
    for i = 1:N
        iteratorids[i] = SSAValue(nextid)
        nextid = addstmt!(destruct,
                          Expr(:call, GlobalRef(Main.Base, :getindex), SSAValue(indicesid), i),
                          tt.parameters[i],
                          nextid)
    end
    # Build the loops
    startloc, stoploc = src.codelocs[id], src.codelocs[bodystop+1]
    loopbody = PrePostBlock(StmtBlock(startloc),
                            body,
                            StmtBlock(stoploc))
    iteritem_ids  = Array{SSAValue}(undef, N)
    exitid = loopexit
    # From the standpoint of marking entrances and exits, it's best to
    # generate the loops from outer-to-inner. But nesting should go
    # the other direction, so we temporarily store a list of
    # independent loops.
    loops = Vector{PrePostBlock}(undef, N)
    for i = N:-1:1
        # Add the `iterate(iter)` call
        pre, enterid, item, itemtype, state, statetype, nextid =
            iterate_block!(src, iteratorids[i],
                           tt.parameters[i],
                           length(src.slotnames)+1,
                           Int(startloc), -1, Int(exitid), Int(nextid))
        iteritem_ids[i]  = item
        # Add the `iterate(iter, state)` call
        post, _, _, _, _, _, nextid =
            iterate_block!(src, iteratorids[i],
                           tt.parameters[i],
                           length(src.slotnames),
                           Int(stoploc), Int(enterid), Int(exitid),
                           Int(nextid), state, statetype)
        loops[i] = PrePostBlock(pre, nothing, post)
        exitid = post.ssavalueids[1]
    end
    # In the loop body, create the `CartesianIndex` that would have
    # resulted from iterating the CartesianIndices
    tupleid = SSAValue(nextid)
    nextid = addstmt!(loopbody.pre,
                      Expr(:call, GlobalRef(Core, :tuple), iteritem_ids...),
                      NTuple{N,Int},
                      nextid)
    nextid = addstmt!(loopbody.pre,
                      Expr(:(=), itervar, Expr(:new, Main.Base.CartesianIndex{N}, tupleid)),
                      Main.Base.CartesianIndex{N},
                      nextid)
    # Nest the loops
    outerloop = loop = loops[N]
    for i = N-1:-1:1
        loop.body = loops[i]
        loop = loops[i]
    end
    loop.body = loopbody
    # Concatenate back to a linear representation
    whole = StmtBlock(wholepre.codelocdefault)
    append!(whole, wholepre)
    append!(whole, destruct)
    append!(whole, outerloop)
    append!(whole, wholepost)
    # Fix up the numbering
    changemap = fill(typemin(Int), nextid)
    for i = 1:length(whole.ssavalueids)
        id = whole.ssavalueids[i]
        changemap[id] = i - id
    end
    remap_ir_elements!(whole.stmts, changemap)
    src.code = whole.stmts
    src.codelocs = whole.codelocs
    src.ssavaluetypes = whole.ssavaluetypes
    return src
    # newsrc = CodeInfo(whole.stmts, whole.codelocs, whole.ssavaluetypes,
    #                   src)
    # println("Old code:\n", src)
    # println("New code:\n", newrc)
    # return src
end


function expand_cartesian_loops!(src::CodeInfo)
    if isdefined(Main, :Base) && isdefined(Main.Base, :CartesianIndices)
        modified = false
        for i = 1:length(src.code)
            rhs = getrhs(src, i)
            if isexpr(rhs, :call) && length(rhs.args) == 2
                if isglobalref(rhs.args[1], Main.Base, :iterate)
                    iterator = rhs.args[2]
                    iteratortype = if isa(iterator, SSAValue)
                        src.ssavaluetypes[iterator.id]
                    elseif isa(iterator, SlotNumber) || isa(iterator, TypedSlot)
                        src.slottypes[iterator.id]
                    else
                        Any
                    end
                    if isa(iteratortype, Type) && iteratortype <: Main.Base.CartesianIndices
                        expand_cartesian_loop!(src::CodeInfo, i)
                        modified = true
                        break
                    end
                end
            end
        end
        if modified
            return expand_cartesian_loops!(src)
        end
    end
    return nothing
end
