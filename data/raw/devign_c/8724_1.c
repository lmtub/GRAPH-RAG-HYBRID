static uint16_t nvme_del_sq(NvmeCtrl *n, NvmeCmd *cmd)

{

    NvmeDeleteQ *c = (NvmeDeleteQ *)cmd;

    NvmeRequest *req, *next;

    NvmeSQueue *sq;

    NvmeCQueue *cq;

    uint16_t qid = le16_to_cpu(c->qid);



    if (!qid || nvme_check_sqid(n, qid)) {

        return NVME_INVALID_QID | NVME_DNR;

    }



    sq = n->sq[qid];

    while (!QTAILQ_EMPTY(&sq->out_req_list)) {

        req = QTAILQ_FIRST(&sq->out_req_list);

        assert(req->aiocb);

        blk_aio_cancel(req->aiocb);

    }

    if (!nvme_check_cqid(n, sq->cqid)) {

        cq = n->cq[sq->cqid];

        QTAILQ_REMOVE(&cq->sq_list, sq, entry);



        nvme_post_cqes(cq);

        QTAILQ_FOREACH_SAFE(req, &cq->req_list, entry, next) {

            if (req->sq == sq) {

                QTAILQ_REMOVE(&cq->req_list, req, entry);

                QTAILQ_INSERT_TAIL(&sq->req_list, req, entry);

            }

        }

    }



    nvme_free_sq(sq, n);

    return NVME_SUCCESS;

}
