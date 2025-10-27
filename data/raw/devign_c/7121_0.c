static void virtio_blk_reset(VirtIODevice *vdev)

{

    VirtIOBlock *s = VIRTIO_BLK(vdev);

    AioContext *ctx;

    VirtIOBlockReq *req;



    ctx = blk_get_aio_context(s->blk);

    aio_context_acquire(ctx);

    blk_drain(s->blk);



    /* We drop queued requests after blk_drain() because blk_drain() itself can

     * produce them. */

    while (s->rq) {

        req = s->rq;

        s->rq = req->next;

        virtqueue_detach_element(req->vq, &req->elem, 0);

        virtio_blk_free_request(req);

    }



    if (s->dataplane) {

        virtio_blk_data_plane_stop(s->dataplane);

    }

    aio_context_release(ctx);



    blk_set_enable_write_cache(s->blk, s->original_wce);

}
