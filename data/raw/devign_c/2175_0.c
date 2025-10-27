static int nbd_co_receive_reply(NBDClientSession *s,

                                NBDRequest *request,

                                QEMUIOVector *qiov)

{

    int ret;

    int i = HANDLE_TO_INDEX(s, request->handle);



    /* Wait until we're woken up by nbd_read_reply_entry.  */

    s->requests[i].receiving = true;

    qemu_coroutine_yield();

    s->requests[i].receiving = false;

    if (s->reply.handle != request->handle || !s->ioc || s->quit) {

        ret = -EIO;

    } else {

        ret = -s->reply.error;

        if (qiov && s->reply.error == 0) {

            assert(request->len == iov_size(qiov->iov, qiov->niov));

            if (qio_channel_readv_all(s->ioc, qiov->iov, qiov->niov,

                                      NULL) < 0) {

                ret = -EIO;

                s->quit = true;

            }

        }



        /* Tell the read handler to read another header.  */

        s->reply.handle = 0;

    }



    s->requests[i].coroutine = NULL;



    /* Kick the read_reply_co to get the next reply.  */

    if (s->read_reply_co) {

        aio_co_wake(s->read_reply_co);

    }



    qemu_co_mutex_lock(&s->send_mutex);

    s->in_flight--;

    qemu_co_queue_next(&s->free_sema);

    qemu_co_mutex_unlock(&s->send_mutex);



    return ret;

}
