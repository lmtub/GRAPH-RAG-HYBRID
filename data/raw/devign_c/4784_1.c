static int nbd_co_receive_request(NBDRequest *req, struct nbd_request *request)

{

    NBDClient *client = req->client;

    int csock = client->sock;

    int rc;



    client->recv_coroutine = qemu_coroutine_self();

    if (nbd_receive_request(csock, request) == -1) {

        rc = -EIO;

        goto out;

    }



    if (request->len > NBD_BUFFER_SIZE) {

        LOG("len (%u) is larger than max len (%u)",

            request->len, NBD_BUFFER_SIZE);

        rc = -EINVAL;

        goto out;

    }



    if ((request->from + request->len) < request->from) {

        LOG("integer overflow detected! "

            "you're probably being attacked");

        rc = -EINVAL;

        goto out;

    }



    TRACE("Decoding type");



    if ((request->type & NBD_CMD_MASK_COMMAND) == NBD_CMD_WRITE) {

        TRACE("Reading %u byte(s)", request->len);



        if (qemu_co_recv(csock, req->data, request->len) != request->len) {

            LOG("reading from socket failed");

            rc = -EIO;

            goto out;

        }

    }

    rc = 0;



out:

    client->recv_coroutine = NULL;

    return rc;

}
