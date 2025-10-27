static struct ioreq *ioreq_start(struct XenBlkDev *blkdev)

{

    struct ioreq *ioreq = NULL;



    if (QLIST_EMPTY(&blkdev->freelist)) {

        if (blkdev->requests_total >= max_requests) {

            goto out;

        }

        /* allocate new struct */

        ioreq = g_malloc0(sizeof(*ioreq));

        ioreq->blkdev = blkdev;

        blkdev->requests_total++;

        qemu_iovec_init(&ioreq->v, BLKIF_MAX_SEGMENTS_PER_REQUEST);

    } else {

        /* get one from freelist */

        ioreq = QLIST_FIRST(&blkdev->freelist);

        QLIST_REMOVE(ioreq, list);

        qemu_iovec_reset(&ioreq->v);

    }

    QLIST_INSERT_HEAD(&blkdev->inflight, ioreq, list);

    blkdev->requests_inflight++;



out:

    return ioreq;

}
