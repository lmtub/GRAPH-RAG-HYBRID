static void bdrv_co_drain_bh_cb(void *opaque)

{

    BdrvCoDrainData *data = opaque;

    Coroutine *co = data->co;

    BlockDriverState *bs = data->bs;



    bdrv_dec_in_flight(bs);

    bdrv_drain_poll(bs);

    data->done = true;

    qemu_coroutine_enter(co);

}
