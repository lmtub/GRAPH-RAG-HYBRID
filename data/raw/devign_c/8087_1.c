int net_init_bridge(const NetClientOptions *opts, const char *name,

                    NetClientState *peer, Error **errp)

{

    /* FIXME error_setg(errp, ...) on failure */

    const NetdevBridgeOptions *bridge;

    const char *helper, *br;

    TAPState *s;

    int fd, vnet_hdr;



    assert(opts->kind == NET_CLIENT_OPTIONS_KIND_BRIDGE);

    bridge = opts->bridge;



    helper = bridge->has_helper ? bridge->helper : DEFAULT_BRIDGE_HELPER;

    br     = bridge->has_br     ? bridge->br     : DEFAULT_BRIDGE_INTERFACE;



    fd = net_bridge_run_helper(helper, br);

    if (fd == -1) {

        return -1;

    }



    fcntl(fd, F_SETFL, O_NONBLOCK);

    vnet_hdr = tap_probe_vnet_hdr(fd);

    s = net_tap_fd_init(peer, "bridge", name, fd, vnet_hdr);



    snprintf(s->nc.info_str, sizeof(s->nc.info_str), "helper=%s,br=%s", helper,

             br);



    return 0;

}
