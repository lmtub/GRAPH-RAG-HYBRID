static void paio_cancel(BlockDriverAIOCB *blockacb)

{

    struct qemu_paiocb *acb = (struct qemu_paiocb *)blockacb;

    int active = 0;



    mutex_lock(&lock);

    if (!acb->active) {

        TAILQ_REMOVE(&request_list, acb, node);

        acb->ret = -ECANCELED;

    } else if (acb->ret == -EINPROGRESS) {

        active = 1;

    }

    mutex_unlock(&lock);



    if (active) {

        /* fail safe: if the aio could not be canceled, we wait for

           it */

        while (qemu_paio_error(acb) == EINPROGRESS)

            ;

    }



    paio_remove(acb);

}
