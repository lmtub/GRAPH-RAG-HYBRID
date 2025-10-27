static int net_init_nic(QemuOpts *opts, const char *name, VLANState *vlan)

{

    int idx;

    NICInfo *nd;

    const char *netdev;



    idx = nic_get_free_idx();

    if (idx == -1 || nb_nics >= MAX_NICS) {

        error_report("Too Many NICs");

        return -1;

    }



    nd = &nd_table[idx];



    memset(nd, 0, sizeof(*nd));



    if ((netdev = qemu_opt_get(opts, "netdev"))) {

        nd->netdev = qemu_find_netdev(netdev);

        if (!nd->netdev) {

            error_report("netdev '%s' not found", netdev);

            return -1;

        }

    } else {

        assert(vlan);

        nd->vlan = vlan;

    }

    if (name) {

        nd->name = g_strdup(name);

    }

    if (qemu_opt_get(opts, "model")) {

        nd->model = g_strdup(qemu_opt_get(opts, "model"));

    }

    if (qemu_opt_get(opts, "addr")) {

        nd->devaddr = g_strdup(qemu_opt_get(opts, "addr"));

    }



    if (qemu_opt_get(opts, "macaddr") &&

        net_parse_macaddr(nd->macaddr.a, qemu_opt_get(opts, "macaddr")) < 0) {

        error_report("invalid syntax for ethernet address");

        return -1;

    }

    qemu_macaddr_default_if_unset(&nd->macaddr);



    nd->nvectors = qemu_opt_get_number(opts, "vectors",

                                       DEV_NVECTORS_UNSPECIFIED);

    if (nd->nvectors != DEV_NVECTORS_UNSPECIFIED &&

        (nd->nvectors < 0 || nd->nvectors > 0x7ffffff)) {

        error_report("invalid # of vectors: %d", nd->nvectors);

        return -1;

    }



    nd->used = 1;

    nb_nics++;



    return idx;

}
