static ssize_t nbd_co_send_reply(NBDRequest *req, struct nbd_reply *reply,

                                 int len)

{

    NBDClient *client = req->client;

    int csock = client->sock;

    ssize_t rc, ret;



    qemu_co_mutex_lock(&client->send_lock);

    qemu_set_fd_handler2(csock, nbd_can_read, nbd_read,

                         nbd_restart_write, client);

    client->send_coroutine = qemu_coroutine_self();



    if (!len) {

        rc = nbd_send_reply(csock, reply);

    } else {

        socket_set_cork(csock, 1);

        rc = nbd_send_reply(csock, reply);

        if (rc >= 0) {

            ret = qemu_co_send(csock, req->data, len);

            if (ret != len) {

                rc = -EIO;

            }

        }

        socket_set_cork(csock, 0);

    }



    client->send_coroutine = NULL;

    qemu_set_fd_handler2(csock, nbd_can_read, nbd_read, NULL, client);

    qemu_co_mutex_unlock(&client->send_lock);

    return rc;

}
