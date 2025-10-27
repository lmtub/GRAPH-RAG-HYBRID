static void posix_aio_read(void *opaque)

{

    PosixAioState *s = opaque;

    RawAIOCB *acb, **pacb;

    int ret;

    ssize_t len;



    /* read all bytes from signal pipe */

    for (;;) {

        char bytes[16];



        len = read(s->rfd, bytes, sizeof(bytes));

        if (len == -1 && errno == EINTR)

            continue; /* try again */

        if (len == sizeof(bytes))

            continue; /* more to read */

        break;

    }



    for(;;) {

        pacb = &s->first_aio;

        for(;;) {

            acb = *pacb;

            if (!acb)

                goto the_end;

            ret = aio_error(&acb->aiocb);

            if (ret == ECANCELED) {

                /* remove the request */

                *pacb = acb->next;

                raw_fd_pool_put(acb);

                qemu_aio_release(acb);

            } else if (ret != EINPROGRESS) {

                /* end of aio */

                if (ret == 0) {

                    ret = aio_return(&acb->aiocb);

                    if (ret == acb->aiocb.aio_nbytes)

                        ret = 0;

                    else

                        ret = -EINVAL;

                } else {

                    ret = -ret;

                }

                /* remove the request */

                *pacb = acb->next;

                /* call the callback */

                acb->common.cb(acb->common.opaque, ret);

                raw_fd_pool_put(acb);

                qemu_aio_release(acb);

                break;

            } else {

                pacb = &acb->next;

            }

        }

    }

 the_end: ;

}
