int coroutine_fn bdrv_is_allocated(BlockDriverState *bs, int64_t sector_num,

                                   int nb_sectors, int *pnum)

{

    return bdrv_get_block_status(bs, sector_num, nb_sectors, pnum);

}
