static void qemu_laio_process_completion(struct qemu_laio_state *s,

    struct qemu_laiocb *laiocb)

{

    int ret;



    s->count--;



    ret = laiocb->ret;

    if (ret != -ECANCELED) {

        if (ret == laiocb->nbytes)

            ret = 0;

        else if (ret >= 0)

            ret = -EINVAL;



        laiocb->common.cb(laiocb->common.opaque, ret);

    }



    qemu_aio_release(laiocb);

}
