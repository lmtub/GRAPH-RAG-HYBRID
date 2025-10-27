static inline bool bdrv_req_is_aligned(BlockDriverState *bs,

                                       int64_t offset, size_t bytes)

{

    int64_t align = bdrv_get_align(bs);

    return !(offset & (align - 1) || (bytes & (align - 1)));

}
