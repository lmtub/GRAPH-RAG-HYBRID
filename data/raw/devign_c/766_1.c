static TAPState *net_tap_fd_init(VLANState *vlan,

                                 const char *model,

                                 const char *name,

                                 int fd)

{

    TAPState *s;



    s = qemu_mallocz(sizeof(TAPState));

    s->fd = fd;

    s->vc = qemu_new_vlan_client(vlan, model, name, tap_receive, NULL, s);

    s->vc->fd_readv = tap_receive_iov;

    qemu_set_fd_handler(s->fd, tap_send, NULL, s);

    snprintf(s->vc->info_str, sizeof(s->vc->info_str), "fd=%d", fd);

    return s;

}
