static void ioreq_release(struct ioreq *ioreq)

{

    struct XenBlkDev *blkdev = ioreq->blkdev;



    LIST_REMOVE(ioreq, list);

    memset(ioreq, 0, sizeof(*ioreq));

    ioreq->blkdev = blkdev;

    LIST_INSERT_HEAD(&blkdev->freelist, ioreq, list);

    blkdev->requests_finished--;

}
