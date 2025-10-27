static int ioreq_runio_qemu_aio(struct ioreq *ioreq)

{

    struct XenBlkDev *blkdev = ioreq->blkdev;



    if (ioreq->req.nr_segments && ioreq_map(ioreq) == -1) {

        goto err_no_map;

    }



    ioreq->aio_inflight++;

    if (ioreq->presync) {

        bdrv_aio_flush(ioreq->blkdev->bs, qemu_aio_complete, ioreq);

        return 0;

    }



    switch (ioreq->req.operation) {

    case BLKIF_OP_READ:

        block_acct_start(bdrv_get_stats(blkdev->bs), &ioreq->acct,

                         ioreq->v.size, BLOCK_ACCT_READ);

        ioreq->aio_inflight++;

        bdrv_aio_readv(blkdev->bs, ioreq->start / BLOCK_SIZE,

                       &ioreq->v, ioreq->v.size / BLOCK_SIZE,

                       qemu_aio_complete, ioreq);

        break;

    case BLKIF_OP_WRITE:

    case BLKIF_OP_FLUSH_DISKCACHE:

        if (!ioreq->req.nr_segments) {

            break;

        }



        block_acct_start(bdrv_get_stats(blkdev->bs), &ioreq->acct,

                         ioreq->v.size, BLOCK_ACCT_WRITE);

        ioreq->aio_inflight++;

        bdrv_aio_writev(blkdev->bs, ioreq->start / BLOCK_SIZE,

                        &ioreq->v, ioreq->v.size / BLOCK_SIZE,

                        qemu_aio_complete, ioreq);

        break;

    case BLKIF_OP_DISCARD:

    {

        struct blkif_request_discard *discard_req = (void *)&ioreq->req;

        ioreq->aio_inflight++;

        bdrv_aio_discard(blkdev->bs,

                        discard_req->sector_number, discard_req->nr_sectors,

                        qemu_aio_complete, ioreq);

        break;

    }

    default:

        /* unknown operation (shouldn't happen -- parse catches this) */

        goto err;

    }



    qemu_aio_complete(ioreq, 0);



    return 0;



err:

    ioreq_unmap(ioreq);

err_no_map:

    ioreq_finish(ioreq);

    ioreq->status = BLKIF_RSP_ERROR;

    return -1;

}
