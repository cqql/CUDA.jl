# interfacing with other packages

using LinearAlgebra
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal

function mv_wrapper(transa::SparseChar, alpha::Number, A::CuSparseMatrix{T}, X::DenseCuVector{T},
                    beta::Number, Y::CuVector{T}) where {T}
    mv!(transa, alpha, A, X, beta, Y, 'O')
end

function mm_wrapper(transa::SparseChar, transb::SparseChar, alpha::Number,
                    A::CuSparseMatrix{T}, B::CuMatrix{T}, beta::Number, C::CuMatrix{T}) where {T}
    if version() <= v"10.3.1"
        # Generic mm! doesn't support transposed B on CUDA10
        return mm2!(transa, transb, alpha, A, B, beta, C, 'O')
    end
    mm!(transa, transb, alpha, A, B, beta, C, 'O')
end

tag_wrappers = ((identity, identity),
                (T -> :(HermOrSym{T, <:$T}), A -> :(parent($A))))
op_wrappers = (
    (identity, 'N', identity),
    (T -> :(Transpose{<:T, <:$T}), 'T', A -> :(parent($A))),
    (T -> :(Adjoint{<:T, <:$T}), 'C', A -> :(parent($A)))
)
for (taga, untaga) in tag_wrappers, (wrapa, transa, unwrapa) in op_wrappers
    TypeA = wrapa(taga(:(CuSparseMatrix{T})))

    @eval begin
        function LinearAlgebra.mul!(C::CuVector{T}, A::$TypeA, B::DenseCuVector{T},
                                    alpha::Number, beta::Number) where {T <: BlasFloat}
            mv_wrapper($transa, alpha, $(untaga(unwrapa(:A))), B, beta, C)
        end
    end

    for (tagb, untagb) in tag_wrappers, (wrapb, transb, unwrapb) in op_wrappers
        TypeB = wrapb(tagb(:(DenseCuMatrix{T})))

        isadjoint(expr) = expr.head == :curly && expr.args[1] == :Adjoint
        if isadjoint(TypeA) || isadjoint(TypeB)
            # CUSPARSE defines adjoints only for complex inputs. For real inputs we run
            # adjoints as tranposes so that we can still support the whole API surface of
            # LinearAlgebra.
            @eval begin
                function LinearAlgebra.mul!(C::CuMatrix{T}, A::$TypeA, B::$TypeB,
                                            alpha::Number, beta::Number) where {T <: BlasComplex}
                    mm_wrapper($transa, $transb, alpha, $(untaga(unwrapa(:A))),
                               $(untagb(unwrapb(:B))), beta, C)
                end
            end

            transa_real = transa == 'C' ? 'T' : transa
            transb_real = transb == 'C' ? 'T' : transb
            @eval begin
                function LinearAlgebra.mul!(C::CuMatrix{T}, A::$TypeA, B::$TypeB,
                                            alpha::Number, beta::Number) where {T <: BlasReal}
                    mm_wrapper($transa_real, $transb_real, alpha, $(untaga(unwrapa(:A))),
                               $(untagb(unwrapb(:B))), beta, C)
                end
            end
        else
            @eval begin
                function LinearAlgebra.mul!(C::CuMatrix{T}, A::$TypeA, B::$TypeB,
                                            alpha::Number, beta::Number) where {T <: BlasFloat}
                    mm_wrapper($transa, $transb, alpha, $(untaga(unwrapa(:A))),
                               $(untagb(unwrapb(:B))), beta, C)
                end
            end
        end
    end
end

Base.:(\)(A::Union{UpperTriangular{T, S},LowerTriangular{T, S}}, B::DenseCuMatrix{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sm('N',A,B,'O')
Base.:(\)(transA::Transpose{T, UpperTriangular{T, S}}, B::DenseCuMatrix{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sm('T',parent(transA),B,'O')
Base.:(\)(transA::Transpose{T, LowerTriangular{T, S}}, B::DenseCuMatrix{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sm('T',parent(transA),B,'O')
Base.:(\)(adjA::Adjoint{T, UpperTriangular{T, S}},B::DenseCuMatrix{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sm('C',parent(adjA),B,'O')
Base.:(\)(adjA::Adjoint{T, LowerTriangular{T, S}},B::DenseCuMatrix{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sm('C',parent(adjA),B,'O')

Base.:(\)(A::Union{UpperTriangular{T, S},LowerTriangular{T, S}}, B::DenseCuVector{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}}       = sv2('N',A,B,'O')
Base.:(\)(transA::Transpose{T, UpperTriangular{T, S}},B::DenseCuVector{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sv2('T',parent(transA),B,'O')
Base.:(\)(transA::Transpose{T, LowerTriangular{T, S}},B::DenseCuVector{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sv2('T',parent(transA),B,'O')
Base.:(\)(adjA::Adjoint{T, UpperTriangular{T, S}},B::DenseCuVector{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}}  = sv2('C',parent(adjA),B,'O')
Base.:(\)(adjA::Adjoint{T, LowerTriangular{T, S}},B::DenseCuVector{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}}  = sv2('C',parent(adjA),B,'O')

Base.:(+)(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC},B::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}) = geam(A,B,'O','O','O')
Base.:(-)(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC},B::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}) = geam(A,-one(eltype(A)),B,'O','O','O')
