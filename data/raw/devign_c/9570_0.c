static void net_socket_update_fd_handler(NetSocketState *s)

{

    qemu_set_fd_handler2(s->fd,

                         s->read_poll  ? net_socket_can_send : NULL,

                         s->read_poll  ? s->send_fn : NULL,

                         s->write_poll ? net_socket_writable : NULL,

                         s);

}
