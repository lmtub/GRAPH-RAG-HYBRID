static VirtIOBlockReq *virtio_blk_alloc_request(VirtIOBlock *s)

{

    VirtIOBlockReq *req = g_slice_new(VirtIOBlockReq);

    req->dev = s;

    req->qiov.size = 0;

    req->next = NULL;

    req->elem = g_slice_new(VirtQueueElement);

    return req;

}
