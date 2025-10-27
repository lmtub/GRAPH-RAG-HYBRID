static int nbd_co_send_request(BDRVNBDState *s, struct nbd_request *request,

                               struct iovec *iov, int offset)

{

    int rc, ret;



    qemu_co_mutex_lock(&s->send_mutex);

    s->send_coroutine = qemu_coroutine_self();

    qemu_aio_set_fd_handler(s->sock, nbd_reply_ready, nbd_restart_write,

                            nbd_have_request, NULL, s);

    rc = nbd_send_request(s->sock, request);

    if (rc >= 0 && iov) {

        ret = qemu_co_sendv(s->sock, iov, request->len, offset);

        if (ret != request->len) {

            errno = -EIO;

            rc = -1;

        }

    }

    qemu_aio_set_fd_handler(s->sock, nbd_reply_ready, NULL,

                            nbd_have_request, NULL, s);

    s->send_coroutine = NULL;

    qemu_co_mutex_unlock(&s->send_mutex);

    return rc;

}
