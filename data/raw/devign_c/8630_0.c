static int net_vde_init(VLANState *vlan, const char *model,

                        const char *name, const char *sock,

                        int port, const char *group, int mode)

{

    VDEState *s;

    char *init_group = strlen(group) ? (char *)group : NULL;

    char *init_sock = strlen(sock) ? (char *)sock : NULL;



    struct vde_open_args args = {

        .port = port,

        .group = init_group,

        .mode = mode,

    };



    s = qemu_mallocz(sizeof(VDEState));

    s->vde = vde_open(init_sock, (char *)"QEMU", &args);

    if (!s->vde){

        free(s);

        return -1;

    }

    s->vc = qemu_new_vlan_client(vlan, model, name, NULL, vde_from_qemu,

                                 NULL, vde_cleanup, s);

    qemu_set_fd_handler(vde_datafd(s->vde), vde_to_qemu, NULL, s);

    snprintf(s->vc->info_str, sizeof(s->vc->info_str), "sock=%s,fd=%d",

             sock, vde_datafd(s->vde));

    return 0;

}
