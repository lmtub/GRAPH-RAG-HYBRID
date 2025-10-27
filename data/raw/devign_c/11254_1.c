static int coroutine_fn raw_co_write_zeroes(

    BlockDriverState *bs, int64_t sector_num,

    int nb_sectors, BdrvRequestFlags flags)

{

    BDRVRawState *s = bs->opaque;



    if (!(flags & BDRV_REQ_MAY_UNMAP)) {

        return -ENOTSUP;

    }

    if (!s->discard_zeroes) {

        return -ENOTSUP;

    }

    return paio_submit_co(bs, s->fd, sector_num, NULL, nb_sectors,

                          QEMU_AIO_DISCARD);

}
