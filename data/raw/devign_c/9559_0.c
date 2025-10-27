static void tap_update_fd_handler(TAPState *s)

{

    qemu_set_fd_handler2(s->fd,

                         s->read_poll && s->enabled ? tap_can_send : NULL,

                         s->read_poll && s->enabled ? tap_send     : NULL,

                         s->write_poll && s->enabled ? tap_writable : NULL,

                         s);

}
