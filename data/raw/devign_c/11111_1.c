static int ioreq_runio_qemu_sync(struct ioreq *ioreq)

{

    struct XenBlkDev *blkdev = ioreq->blkdev;

    int i, rc, len = 0;

    off_t pos;



    if (ioreq->req.nr_segments && ioreq_map(ioreq) == -1)

	goto err;

    if (ioreq->presync)

	bdrv_flush(blkdev->bs);



    switch (ioreq->req.operation) {

    case BLKIF_OP_READ:

	pos = ioreq->start;

	for (i = 0; i < ioreq->v.niov; i++) {

	    rc = bdrv_read(blkdev->bs, pos / BLOCK_SIZE,

			   ioreq->v.iov[i].iov_base,

			   ioreq->v.iov[i].iov_len / BLOCK_SIZE);

	    if (rc != 0) {

		xen_be_printf(&blkdev->xendev, 0, "rd I/O error (%p, len %zd)\n",

			      ioreq->v.iov[i].iov_base,

			      ioreq->v.iov[i].iov_len);

		goto err;

	    }

	    len += ioreq->v.iov[i].iov_len;

	    pos += ioreq->v.iov[i].iov_len;

	}

	break;

    case BLKIF_OP_WRITE:

    case BLKIF_OP_WRITE_BARRIER:

        if (!ioreq->req.nr_segments)

            break;

	pos = ioreq->start;

	for (i = 0; i < ioreq->v.niov; i++) {

	    rc = bdrv_write(blkdev->bs, pos / BLOCK_SIZE,

			    ioreq->v.iov[i].iov_base,

			    ioreq->v.iov[i].iov_len / BLOCK_SIZE);

	    if (rc != 0) {

		xen_be_printf(&blkdev->xendev, 0, "wr I/O error (%p, len %zd)\n",

			      ioreq->v.iov[i].iov_base,

			      ioreq->v.iov[i].iov_len);

		goto err;

	    }

	    len += ioreq->v.iov[i].iov_len;

	    pos += ioreq->v.iov[i].iov_len;

	}

	break;

    default:

	/* unknown operation (shouldn't happen -- parse catches this) */

	goto err;

    }



    if (ioreq->postsync)

	bdrv_flush(blkdev->bs);

    ioreq->status = BLKIF_RSP_OKAY;



    ioreq_unmap(ioreq);

    ioreq_finish(ioreq);

    return 0;



err:

    ioreq->status = BLKIF_RSP_ERROR;

    return -1;

}
