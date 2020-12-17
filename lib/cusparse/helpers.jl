# cuSPARSE helper functions


## matrix descriptor

mutable struct CuMatrixDescriptor
    handle::cusparseMatDescr_t

    function CuMatrixDescriptor()
        descr_ref = Ref{cusparseMatDescr_t}()
        cusparseCreateMatDescr(descr_ref)
        obj = new(descr_ref[])
        finalizer(cusparseDestroyMatDescr, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusparseMatDescr_t}, desc::CuMatrixDescriptor) = desc.handle

function CuMatrixDescriptor(MatrixType, FillMode, DiagType, IndexBase)
    desc = CuMatrixDescriptor()
    if MatrixType != CUSPARSE_MATRIX_TYPE_GENERAL
        cusparseSetMatType(desc, MatrixType)
    end
    cusparseSetMatFillMode(desc, FillMode)
    cusparseSetMatDiagType(desc, DiagType)
    if IndexBase != CUSPARSE_INDEX_BASE_ZERO
        cusparseSetMatIndexBase(desc, IndexBase)
    end
    return desc
end


mutable struct CuCsrsv2Info
    handle::csrsv2Info_t

    function CuCsrsv2Info()
        info = @argout cusparseCreateCsrsv2Info(out(Ref{csrsv2Info_t}()))
        obj = new(info[])
        finalizer(cusparseDestroyCsrsv2Info, obj)
        obj
    end
end
Base.unsafe_convert(::Type{csrsm2Info_t}, info::CuCsrsv2Info) = info.handle

mutable struct CuCsrsm2Info
    handle::csrsm2Info_t

    function CuCsrsm2Info()
        info = @argout cusparseCreateCsrsm2Info(out(Ref{csrsm2Info_t}()))
        obj = new(info[])
        finalizer(cusparseDestroyCsrsm2Info, obj)
        obj
    end
end
Base.unsafe_convert(::Type{csrsm2Info_t}, info::CuCsrsm2Info) = info.handle
