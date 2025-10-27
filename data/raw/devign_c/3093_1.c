static void nbd_restart_write(void *opaque)

{

    BlockDriverState *bs = opaque;



    qemu_coroutine_enter(nbd_get_client_session(bs)->send_coroutine, NULL);

}
