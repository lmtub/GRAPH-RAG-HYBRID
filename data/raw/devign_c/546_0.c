static void raw_aio_unplug(BlockDriverState *bs)

{

#ifdef CONFIG_LINUX_AIO

    BDRVRawState *s = bs->opaque;

    if (s->use_aio) {

        laio_io_unplug(bs, s->aio_ctx, true);

    }

#endif

}
