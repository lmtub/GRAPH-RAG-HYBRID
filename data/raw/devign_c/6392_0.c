static void ioreq_unmap(struct ioreq *ioreq)

{

    int gnt = ioreq->blkdev->xendev.gnttabdev;

    int i;



    if (ioreq->v.niov == 0) {

        return;

    }

    if (batch_maps) {

        if (!ioreq->pages) {

            return;

        }

        if (xc_gnttab_munmap(gnt, ioreq->pages, ioreq->v.niov) != 0) {

            xen_be_printf(&ioreq->blkdev->xendev, 0, "xc_gnttab_munmap failed: %s\n",

                          strerror(errno));

        }

        ioreq->blkdev->cnt_map -= ioreq->v.niov;

        ioreq->pages = NULL;

    } else {

        for (i = 0; i < ioreq->v.niov; i++) {

            if (!ioreq->page[i]) {

                continue;

            }

            if (xc_gnttab_munmap(gnt, ioreq->page[i], 1) != 0) {

                xen_be_printf(&ioreq->blkdev->xendev, 0, "xc_gnttab_munmap failed: %s\n",

                              strerror(errno));

            }

            ioreq->blkdev->cnt_map--;

            ioreq->page[i] = NULL;

        }

    }

}
