static void qemu_rbd_complete_aio(RADOSCB *rcb)

{

    RBDAIOCB *acb = rcb->acb;

    int64_t r;



    r = rcb->ret;



    if (acb->cmd != RBD_AIO_READ) {

        if (r < 0) {

            acb->ret = r;

            acb->error = 1;

        } else if (!acb->error) {

            acb->ret = rcb->size;

        }

    } else {

        if (r < 0) {

            memset(rcb->buf, 0, rcb->size);

            acb->ret = r;

            acb->error = 1;

        } else if (r < rcb->size) {

            memset(rcb->buf + r, 0, rcb->size - r);

            if (!acb->error) {

                acb->ret = rcb->size;

            }

        } else if (!acb->error) {

            acb->ret = r;

        }

    }

    /* Note that acb->bh can be NULL in case where the aio was cancelled */

    acb->bh = qemu_bh_new(rbd_aio_bh_cb, acb);

    qemu_bh_schedule(acb->bh);

    g_free(rcb);

}
