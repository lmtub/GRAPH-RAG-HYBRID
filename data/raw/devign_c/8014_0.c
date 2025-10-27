static int netmap_can_send(void *opaque)

{

    NetmapState *s = opaque;



    return qemu_can_send_packet(&s->nc);

}
