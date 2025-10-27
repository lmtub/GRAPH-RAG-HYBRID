static int bdrv_rd_badreq_bytes(BlockDriverState *bs,

                                int64_t offset, int count)

{

    int64_t size = bs->total_sectors << SECTOR_BITS;

    return

        count < 0 ||

        size < 0 ||

        count > size ||

        offset > size - count;

}
