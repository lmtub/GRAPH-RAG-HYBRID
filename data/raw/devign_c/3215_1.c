void bdrv_close_all(void)

{

    BlockDriverState *bs;



    QTAILQ_FOREACH(bs, &bdrv_states, device_list) {

        AioContext *aio_context = bdrv_get_aio_context(bs);



        aio_context_acquire(aio_context);

        bdrv_close(bs);

        aio_context_release(aio_context);

    }

}
