static int ioreq_parse(struct ioreq *ioreq)

{

    struct XenBlkDev *blkdev = ioreq->blkdev;

    uintptr_t mem;

    size_t len;

    int i;



    xen_be_printf(&blkdev->xendev, 3,

		  "op %d, nr %d, handle %d, id %" PRId64 ", sector %" PRId64 "\n",

		  ioreq->req.operation, ioreq->req.nr_segments,

		  ioreq->req.handle, ioreq->req.id, ioreq->req.sector_number);

    switch (ioreq->req.operation) {

    case BLKIF_OP_READ:

	ioreq->prot = PROT_WRITE; /* to memory */

        if (ioreq->req.operation != BLKIF_OP_READ && blkdev->mode[0] != 'w') {

	    xen_be_printf(&blkdev->xendev, 0, "error: write req for ro device\n");

	    goto err;

	}

	break;

    case BLKIF_OP_WRITE_BARRIER:

	if (!syncwrite)

	    ioreq->presync = ioreq->postsync = 1;

	/* fall through */

    case BLKIF_OP_WRITE:

	ioreq->prot = PROT_READ; /* from memory */

	if (syncwrite)

	    ioreq->postsync = 1;

	break;

    default:

	xen_be_printf(&blkdev->xendev, 0, "error: unknown operation (%d)\n",

		      ioreq->req.operation);

	goto err;

    };



    ioreq->start = ioreq->req.sector_number * blkdev->file_blk;

    for (i = 0; i < ioreq->req.nr_segments; i++) {

	if (i == BLKIF_MAX_SEGMENTS_PER_REQUEST) {

	    xen_be_printf(&blkdev->xendev, 0, "error: nr_segments too big\n");

	    goto err;

	}

	if (ioreq->req.seg[i].first_sect > ioreq->req.seg[i].last_sect) {

	    xen_be_printf(&blkdev->xendev, 0, "error: first > last sector\n");

	    goto err;

	}

	if (ioreq->req.seg[i].last_sect * BLOCK_SIZE >= XC_PAGE_SIZE) {

	    xen_be_printf(&blkdev->xendev, 0, "error: page crossing\n");

	    goto err;

	}



	ioreq->domids[i] = blkdev->xendev.dom;

	ioreq->refs[i]   = ioreq->req.seg[i].gref;



	mem = ioreq->req.seg[i].first_sect * blkdev->file_blk;

	len = (ioreq->req.seg[i].last_sect - ioreq->req.seg[i].first_sect + 1) * blkdev->file_blk;

        qemu_iovec_add(&ioreq->v, (void*)mem, len);

    }

    if (ioreq->start + ioreq->v.size > blkdev->file_size) {

	xen_be_printf(&blkdev->xendev, 0, "error: access beyond end of file\n");

	goto err;

    }

    return 0;



err:

    ioreq->status = BLKIF_RSP_ERROR;

    return -1;

}
