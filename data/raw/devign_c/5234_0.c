static void vde_to_qemu(void *opaque)

{

    VDEState *s = opaque;

    uint8_t buf[4096];

    int size;



    size = vde_recv(s->vde, (char *)buf, sizeof(buf), 0);

    if (size > 0) {

        qemu_send_packet(&s->nc, buf, size);

    }

}
