static int virtio_blk_handle_rw_error(VirtIOBlockReq *req, int error,

    bool is_read)

{

    BlockErrorAction action = blk_get_error_action(req->dev->blk,

                                                   is_read, error);

    VirtIOBlock *s = req->dev;



    if (action == BLOCK_ERROR_ACTION_STOP) {

        req->next = s->rq;

        s->rq = req;

    } else if (action == BLOCK_ERROR_ACTION_REPORT) {

        virtio_blk_req_complete(req, VIRTIO_BLK_S_IOERR);

        block_acct_done(blk_get_stats(s->blk), &req->acct);

        virtio_blk_free_request(req);

    }



    blk_error_action(s->blk, action, is_read, error);

    return action != BLOCK_ERROR_ACTION_IGNORE;

}
