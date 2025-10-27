BlockAIOCB *bdrv_aio_discard(BlockDriverState *bs,

        int64_t sector_num, int nb_sectors,

        BlockCompletionFunc *cb, void *opaque)

{

    Coroutine *co;

    BlockAIOCBCoroutine *acb;



    trace_bdrv_aio_discard(bs, sector_num, nb_sectors, opaque);



    acb = qemu_aio_get(&bdrv_em_co_aiocb_info, bs, cb, opaque);

    acb->need_bh = true;

    acb->req.error = -EINPROGRESS;

    acb->req.sector = sector_num;

    acb->req.nb_sectors = nb_sectors;

    co = qemu_coroutine_create(bdrv_aio_discard_co_entry);

    qemu_coroutine_enter(co, acb);



    bdrv_co_maybe_schedule_bh(acb);

    return &acb->common;

}
