VirtIOBlockReq *virtio_blk_alloc_request(VirtIOBlock *s)

{

    VirtIOBlockReq *req = g_new(VirtIOBlockReq, 1);

    req->dev = s;

    req->qiov.size = 0;

    req->in_len = 0;

    req->next = NULL;

    req->mr_next = NULL;

    return req;

}
