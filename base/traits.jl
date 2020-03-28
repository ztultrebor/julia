# This file is a part of Julia. License is MIT: https://julialang.org/license

## numeric/object traits
# trait for objects that have an ordering
abstract type OrderStyle end
struct Ordered <: OrderStyle end
struct Unordered <: OrderStyle end

OrderStyle(instance) = OrderStyle(typeof(instance))
OrderStyle(::Type{<:Real}) = Ordered()
OrderStyle(::Type{<:Any}) = Unordered()

# trait for objects that support arithmetic
abstract type ArithmeticStyle end
struct ArithmeticRounds <: ArithmeticStyle end     # least significant bits can be lost
struct ArithmeticWraps <: ArithmeticStyle end      #  most significant bits can be lost
struct ArithmeticUnknown <: ArithmeticStyle end

ArithmeticStyle(instance) = ArithmeticStyle(typeof(instance))
ArithmeticStyle(::Type{<:AbstractFloat}) = ArithmeticRounds()
ArithmeticStyle(::Type{<:Integer}) = ArithmeticWraps()
ArithmeticStyle(::Type{<:Any}) = ArithmeticUnknown()
