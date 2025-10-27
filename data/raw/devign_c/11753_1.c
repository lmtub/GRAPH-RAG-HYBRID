static void net_dump_cleanup(VLANClientState *vc)

{

    DumpState *s = vc->opaque;



    close(s->fd);

    qemu_free(s);

}
