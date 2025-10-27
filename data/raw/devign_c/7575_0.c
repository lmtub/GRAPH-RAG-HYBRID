static void nbd_restart_write(void *opaque)

{

    NbdClientSession *s = opaque;



    qemu_coroutine_enter(s->send_coroutine, NULL);

}
