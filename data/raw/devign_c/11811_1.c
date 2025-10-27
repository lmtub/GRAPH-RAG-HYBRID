static int net_init_nic(const NetClientOptions *opts, const char *name,

                        NetClientState *peer, Error **errp)

{

    /* FIXME error_setg(errp, ...) on failure */

    int idx;

    NICInfo *nd;

    const NetLegacyNicOptions *nic;



    assert(opts->kind == NET_CLIENT_OPTIONS_KIND_NIC);

    nic = opts->nic;



    idx = nic_get_free_idx();

    if (idx == -1 || nb_nics >= MAX_NICS) {

        error_report("Too Many NICs");

        return -1;

    }



    nd = &nd_table[idx];



    memset(nd, 0, sizeof(*nd));



    if (nic->has_netdev) {

        nd->netdev = qemu_find_netdev(nic->netdev);

        if (!nd->netdev) {

            error_report("netdev '%s' not found", nic->netdev);

            return -1;

        }

    } else {

        assert(peer);

        nd->netdev = peer;

    }

    nd->name = g_strdup(name);

    if (nic->has_model) {

        nd->model = g_strdup(nic->model);

    }

    if (nic->has_addr) {

        nd->devaddr = g_strdup(nic->addr);

    }



    if (nic->has_macaddr &&

        net_parse_macaddr(nd->macaddr.a, nic->macaddr) < 0) {

        error_report("invalid syntax for ethernet address");

        return -1;

    }

    if (nic->has_macaddr &&

        is_multicast_ether_addr(nd->macaddr.a)) {

        error_report("NIC cannot have multicast MAC address (odd 1st byte)");

        return -1;

    }

    qemu_macaddr_default_if_unset(&nd->macaddr);



    if (nic->has_vectors) {

        if (nic->vectors > 0x7ffffff) {

            error_report("invalid # of vectors: %"PRIu32, nic->vectors);

            return -1;

        }

        nd->nvectors = nic->vectors;

    } else {

        nd->nvectors = DEV_NVECTORS_UNSPECIFIED;

    }



    nd->used = 1;

    nb_nics++;



    return idx;

}
