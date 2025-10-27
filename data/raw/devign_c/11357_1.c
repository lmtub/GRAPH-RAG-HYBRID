static void nbd_read(void *opaque)

{

    NBDClient *client = opaque;



    if (client->recv_coroutine) {

        qemu_coroutine_enter(client->recv_coroutine, NULL);

    } else {

        qemu_coroutine_enter(qemu_coroutine_create(nbd_trip), client);

    }

}
