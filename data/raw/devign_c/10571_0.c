int v9fs_co_readlink(V9fsPDU *pdu, V9fsPath *path, V9fsString *buf)

{

    int err;

    ssize_t len;

    V9fsState *s = pdu->s;



    if (v9fs_request_cancelled(pdu)) {

        return -EINTR;

    }

    buf->data = g_malloc(PATH_MAX);

    v9fs_path_read_lock(s);

    v9fs_co_run_in_worker(

        {

            len = s->ops->readlink(&s->ctx, path,

                                   buf->data, PATH_MAX - 1);

            if (len > -1) {

                buf->size = len;

                buf->data[len] = 0;

                err = 0;

            } else {

                err = -errno;

            }

        });

    v9fs_path_unlock(s);

    if (err) {

        g_free(buf->data);

        buf->data = NULL;

        buf->size = 0;

    }

    return err;

}
