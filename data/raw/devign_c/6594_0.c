static int net_socket_can_send(void *opaque)

{

    NetSocketState *s = opaque;



    return qemu_can_send_packet(&s->nc);

}
