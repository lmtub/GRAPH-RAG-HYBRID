int net_init_netmap(const NetClientOptions *opts,

                    const char *name, NetClientState *peer, Error **errp)

{

    const NetdevNetmapOptions *netmap_opts = opts->u.netmap;

    struct nm_desc *nmd;

    NetClientState *nc;

    Error *err = NULL;

    NetmapState *s;



    nmd = netmap_open(netmap_opts, &err);

    if (err) {

        error_propagate(errp, err);

        return -1;

    }

    /* Create the object. */

    nc = qemu_new_net_client(&net_netmap_info, peer, "netmap", name);

    s = DO_UPCAST(NetmapState, nc, nc);

    s->nmd = nmd;

    s->tx = NETMAP_TXRING(nmd->nifp, 0);

    s->rx = NETMAP_RXRING(nmd->nifp, 0);

    s->vnet_hdr_len = 0;

    pstrcpy(s->ifname, sizeof(s->ifname), netmap_opts->ifname);

    netmap_read_poll(s, true); /* Initially only poll for reads. */



    return 0;

}
