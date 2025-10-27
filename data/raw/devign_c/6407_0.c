static void virtio_blk_dma_restart_bh(void *opaque)

{

    VirtIOBlock *s = opaque;

    VirtIOBlockReq *req = s->rq;

    MultiReqBuffer mrb = {

        .num_writes = 0,

    };



    qemu_bh_delete(s->bh);

    s->bh = NULL;



    s->rq = NULL;



    while (req) {

        virtio_blk_handle_request(req, &mrb);

        req = req->next;

    }



    if (mrb.num_writes > 0) {

        do_multiwrite(s->bs, mrb.blkreq, mrb.num_writes);

    }

}
