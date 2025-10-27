static int posix_aio_process_queue(void *opaque)

{

    PosixAioState *s = opaque;

    struct qemu_paiocb *acb, **pacb;

    int ret;

    int result = 0;



    for(;;) {

        pacb = &s->first_aio;

        for(;;) {

            acb = *pacb;

            if (!acb)

                return result;



            ret = qemu_paio_error(acb);

            if (ret == ECANCELED) {

                /* remove the request */

                *pacb = acb->next;

                qemu_aio_release(acb);

                result = 1;

            } else if (ret != EINPROGRESS) {

                /* end of aio */

                if (ret == 0) {

                    ret = qemu_paio_return(acb);

                    if (ret == acb->aio_nbytes)

                        ret = 0;

                    else

                        ret = -EINVAL;

                } else {

                    ret = -ret;

                }



                trace_paio_complete(acb, acb->common.opaque, ret);



                /* remove the request */

                *pacb = acb->next;

                /* call the callback */

                acb->common.cb(acb->common.opaque, ret);

                qemu_aio_release(acb);

                result = 1;

                break;

            } else {

                pacb = &acb->next;

            }

        }

    }



    return result;

}
