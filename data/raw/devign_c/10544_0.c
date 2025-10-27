int net_init_vde(QemuOpts *opts, Monitor *mon, const char *name, VLANState *vlan)

{

    const char *sock;

    const char *group;

    int port, mode;



    sock  = qemu_opt_get(opts, "sock");

    group = qemu_opt_get(opts, "group");



    port = qemu_opt_get_number(opts, "port", 0);

    mode = qemu_opt_get_number(opts, "mode", 0700);



    if (net_vde_init(vlan, "vde", name, sock, port, group, mode) == -1) {

        return -1;

    }



    if (vlan) {

        vlan->nb_host_devs++;

    }



    return 0;

}
