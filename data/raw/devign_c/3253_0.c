int bdrv_block_status(BlockDriverState *bs, int64_t offset, int64_t bytes,

                      int64_t *pnum, int64_t *map, BlockDriverState **file)

{

    int64_t ret;

    int n;



    assert(QEMU_IS_ALIGNED(offset | bytes, BDRV_SECTOR_SIZE));

    assert(pnum);

    /*

     * The contract allows us to return pnum smaller than bytes, even

     * if the next query would see the same status; we truncate the

     * request to avoid overflowing the driver's 32-bit interface.

     */

    bytes = MIN(bytes, BDRV_REQUEST_MAX_BYTES);

    ret = bdrv_get_block_status_above(bs, backing_bs(bs),

                                      offset >> BDRV_SECTOR_BITS,

                                      bytes >> BDRV_SECTOR_BITS, &n, file);

    if (ret < 0) {

        assert(INT_MIN <= ret);

        *pnum = 0;

        return ret;

    }

    *pnum = n * BDRV_SECTOR_SIZE;

    if (map) {

        *map = ret & BDRV_BLOCK_OFFSET_MASK;

    } else {

        ret &= ~BDRV_BLOCK_OFFSET_VALID;

    }

    return ret & ~BDRV_BLOCK_OFFSET_MASK;

}
