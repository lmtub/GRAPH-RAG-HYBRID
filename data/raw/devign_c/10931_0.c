static int coroutine_fn bdrv_driver_preadv(BlockDriverState *bs,

                                           uint64_t offset, uint64_t bytes,

                                           QEMUIOVector *qiov, int flags)

{

    BlockDriver *drv = bs->drv;

    int64_t sector_num = offset >> BDRV_SECTOR_BITS;

    unsigned int nb_sectors = bytes >> BDRV_SECTOR_BITS;



    assert((offset & (BDRV_SECTOR_SIZE - 1)) == 0);

    assert((bytes & (BDRV_SECTOR_SIZE - 1)) == 0);

    assert((bytes >> BDRV_SECTOR_BITS) <= BDRV_REQUEST_MAX_SECTORS);



    return drv->bdrv_co_readv(bs, sector_num, nb_sectors, qiov);

}
