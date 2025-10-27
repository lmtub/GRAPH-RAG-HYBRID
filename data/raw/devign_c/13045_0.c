int64_t bdrv_get_block_status(BlockDriverState *bs,

                              int64_t sector_num,

                              int nb_sectors, int *pnum)

{

    return bdrv_get_block_status_above(bs, backing_bs(bs),

                                       sector_num, nb_sectors, pnum);

}
