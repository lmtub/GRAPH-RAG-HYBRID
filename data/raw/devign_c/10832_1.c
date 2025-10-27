static void ivshmem_check_version(void *opaque, const uint8_t * buf, int size)

{

    IVShmemState *s = opaque;

    int tmp;

    int64_t version;



    if (!fifo_update_and_get_i64(s, buf, size, &version)) {

        return;

    }



    tmp = qemu_chr_fe_get_msgfd(s->server_chr);

    if (tmp != -1 || version != IVSHMEM_PROTOCOL_VERSION) {

        fprintf(stderr, "incompatible version, you are connecting to a ivshmem-"

                "server using a different protocol please check your setup\n");

        qemu_chr_add_handlers(s->server_chr, NULL, NULL, NULL, s);

        return;

    }



    IVSHMEM_DPRINTF("version check ok, switch to real chardev handler\n");

    qemu_chr_add_handlers(s->server_chr, ivshmem_can_receive, ivshmem_read,

                          NULL, s);

}
