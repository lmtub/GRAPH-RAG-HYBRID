static void nbd_request_put(NBDRequest *req)

{

    NBDClient *client = req->client;



    if (req->data) {

        qemu_vfree(req->data);

    }

    g_slice_free(NBDRequest, req);



    if (client->nb_requests-- == MAX_NBD_REQUESTS) {

        qemu_notify_event();

    }

    nbd_client_put(client);

}
