static int nbd_co_send_request(BlockDriverState *bs,

                               struct nbd_request *request,

                               QEMUIOVector *qiov, int offset)

{

    NbdClientSession *s = nbd_get_client_session(bs);

    AioContext *aio_context;

    int rc, ret;



    qemu_co_mutex_lock(&s->send_mutex);

    s->send_coroutine = qemu_coroutine_self();

    aio_context = bdrv_get_aio_context(bs);

    aio_set_fd_handler(aio_context, s->sock,

                       nbd_reply_ready, nbd_restart_write, bs);

    if (qiov) {

        if (!s->is_unix) {

            socket_set_cork(s->sock, 1);

        }

        rc = nbd_send_request(s->sock, request);

        if (rc >= 0) {

            ret = qemu_co_sendv(s->sock, qiov->iov, qiov->niov,

                                offset, request->len);

            if (ret != request->len) {

                rc = -EIO;

            }

        }

        if (!s->is_unix) {

            socket_set_cork(s->sock, 0);

        }

    } else {

        rc = nbd_send_request(s->sock, request);

    }

    aio_set_fd_handler(aio_context, s->sock, nbd_reply_ready, NULL, bs);

    s->send_coroutine = NULL;

    qemu_co_mutex_unlock(&s->send_mutex);

    return rc;

}
