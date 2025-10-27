BlockDriverAIOCB *bdrv_aio_readv(BlockDriverState *bs, int64_t sector_num,

                                 QEMUIOVector *qiov, int nb_sectors,

                                 BlockDriverCompletionFunc *cb, void *opaque)

{

    BlockDriver *drv = bs->drv;

    BlockDriverAIOCB *ret;



    trace_bdrv_aio_readv(bs, sector_num, nb_sectors, opaque);



    if (!drv)

        return NULL;

    if (bdrv_check_request(bs, sector_num, nb_sectors))

        return NULL;



    ret = drv->bdrv_aio_readv(bs, sector_num, qiov, nb_sectors,

                              cb, opaque);



    if (ret) {

	/* Update stats even though technically transfer has not happened. */

	bs->rd_bytes += (unsigned) nb_sectors * BDRV_SECTOR_SIZE;

	bs->rd_ops ++;

    }



    return ret;

}
