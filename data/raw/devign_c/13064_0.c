static int nbd_can_read(void *opaque)

{

    NBDClient *client = opaque;



    return client->recv_coroutine || client->nb_requests < MAX_NBD_REQUESTS;

}
