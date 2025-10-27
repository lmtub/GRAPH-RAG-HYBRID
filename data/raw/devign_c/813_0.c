BlockAIOCB *bdrv_aio_flush(BlockDriverState *bs,

        BlockCompletionFunc *cb, void *opaque)

{

    trace_bdrv_aio_flush(bs, opaque);



    Coroutine *co;

    BlockAIOCBCoroutine *acb;



    acb = qemu_aio_get(&bdrv_em_co_aiocb_info, bs, cb, opaque);

    acb->need_bh = true;

    acb->req.error = -EINPROGRESS;



    co = qemu_coroutine_create(bdrv_aio_flush_co_entry);

    qemu_coroutine_enter(co, acb);



    bdrv_co_maybe_schedule_bh(acb);

    return &acb->common;

}
