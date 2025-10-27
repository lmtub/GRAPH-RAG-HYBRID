static int blk_free(struct XenDevice *xendev)

{

    struct XenBlkDev *blkdev = container_of(xendev, struct XenBlkDev, xendev);

    struct ioreq *ioreq;



    while (!LIST_EMPTY(&blkdev->freelist)) {

	ioreq = LIST_FIRST(&blkdev->freelist);

        LIST_REMOVE(ioreq, list);

        qemu_iovec_destroy(&ioreq->v);

	qemu_free(ioreq);

    }



    qemu_free(blkdev->params);

    qemu_free(blkdev->mode);

    qemu_free(blkdev->type);

    qemu_free(blkdev->dev);

    qemu_free(blkdev->devtype);

    qemu_bh_delete(blkdev->bh);

    return 0;

}
