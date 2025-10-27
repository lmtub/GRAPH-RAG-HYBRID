int raw_get_aio_fd(BlockDriverState *bs)

{

    BDRVRawState *s;



    if (!bs->drv) {

        return -ENOMEDIUM;

    }



    if (bs->drv == bdrv_find_format("raw")) {

        bs = bs->file;

    }



    /* raw-posix has several protocols so just check for raw_aio_readv */

    if (bs->drv->bdrv_aio_readv != raw_aio_readv) {

        return -ENOTSUP;

    }



    s = bs->opaque;

    if (!s->use_aio) {

        return -ENOTSUP;

    }

    return s->fd;

}
