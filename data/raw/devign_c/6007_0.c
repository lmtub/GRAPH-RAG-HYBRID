static VirtIOBlockReq *virtio_blk_get_request(VirtIOBlock *s)

{

    VirtIOBlockReq *req = virtio_blk_alloc_request(s);



    if (!virtqueue_pop(s->vq, &req->elem)) {

        virtio_blk_free_request(req);

        return NULL;

    }



    return req;

}
