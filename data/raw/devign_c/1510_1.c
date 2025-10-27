static int coroutine_fn sd_co_flush_to_disk(BlockDriverState *bs)

{

    BDRVSheepdogState *s = bs->opaque;

    SheepdogAIOCB *acb;

    AIOReq *aio_req;



    if (s->cache_flags != SD_FLAG_CMD_CACHE) {

        return 0;

    }



    acb = sd_aio_setup(bs, NULL, 0, 0);

    acb->aiocb_type = AIOCB_FLUSH_CACHE;

    acb->aio_done_func = sd_finish_aiocb;



    aio_req = alloc_aio_req(s, acb, vid_to_vdi_oid(s->inode.vdi_id),

                            0, 0, 0, 0, 0);

    QLIST_INSERT_HEAD(&s->inflight_aio_head, aio_req, aio_siblings);

    add_aio_request(s, aio_req, NULL, 0, false, acb->aiocb_type);



    qemu_coroutine_yield();

    return acb->ret;

}
