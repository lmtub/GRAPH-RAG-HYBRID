static void netmap_update_fd_handler(NetmapState *s)

{

    qemu_set_fd_handler2(s->me.fd,

                         s->read_poll  ? netmap_can_send : NULL,

                         s->read_poll  ? netmap_send     : NULL,

                         s->write_poll ? netmap_writable : NULL,

                         s);

}
