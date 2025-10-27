static int coroutine_fn raw_co_discard(BlockDriverState *bs,

                                       int64_t sector_num, int nb_sectors)

{

    return bdrv_co_discard(bs->file->bs, sector_num, nb_sectors);

}
