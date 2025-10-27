static void nvme_rw_cb(void *opaque, int ret)

{

    NvmeRequest *req = opaque;

    NvmeSQueue *sq = req->sq;

    NvmeCtrl *n = sq->ctrl;

    NvmeCQueue *cq = n->cq[sq->cqid];



    block_acct_done(blk_get_stats(n->conf.blk), &req->acct);

    if (!ret) {

        req->status = NVME_SUCCESS;

    } else {

        req->status = NVME_INTERNAL_DEV_ERROR;

    }

    if (req->has_sg) {

        qemu_sglist_destroy(&req->qsg);

    }

    nvme_enqueue_req_completion(cq, req);

}
