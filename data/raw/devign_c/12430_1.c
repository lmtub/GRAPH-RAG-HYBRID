static void virtio_blk_free_request(VirtIOBlockReq *req)

{

    if (req) {

        g_slice_free(VirtQueueElement, req->elem);

        g_slice_free(VirtIOBlockReq, req);

    }

}
