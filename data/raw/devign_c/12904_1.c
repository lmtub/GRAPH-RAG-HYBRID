static int ioreq_runio_qemu_aio(struct ioreq *ioreq)

{

    struct XenBlkDev *blkdev = ioreq->blkdev;



    if (ioreq->req.nr_segments && ioreq_map(ioreq) == -1)

	goto err;



    ioreq->aio_inflight++;

    if (ioreq->presync)

	bdrv_flush(blkdev->bs); /* FIXME: aio_flush() ??? */



    switch (ioreq->req.operation) {

    case BLKIF_OP_READ:

        ioreq->aio_inflight++;

        bdrv_aio_readv(blkdev->bs, ioreq->start / BLOCK_SIZE,

                       &ioreq->v, ioreq->v.size / BLOCK_SIZE,

                       qemu_aio_complete, ioreq);

	break;

    case BLKIF_OP_WRITE:

    case BLKIF_OP_WRITE_BARRIER:

        ioreq->aio_inflight++;

        if (!ioreq->req.nr_segments)

            break;

        bdrv_aio_writev(blkdev->bs, ioreq->start / BLOCK_SIZE,

                        &ioreq->v, ioreq->v.size / BLOCK_SIZE,

                        qemu_aio_complete, ioreq);

	break;

    default:

	/* unknown operation (shouldn't happen -- parse catches this) */

	goto err;

    }



    if (ioreq->postsync)

	bdrv_flush(blkdev->bs); /* FIXME: aio_flush() ??? */

    qemu_aio_complete(ioreq, 0);



    return 0;



err:

    ioreq->status = BLKIF_RSP_ERROR;

    return -1;

}
