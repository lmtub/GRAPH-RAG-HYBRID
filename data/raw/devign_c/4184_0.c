int net_init_tap(const NetClientOptions *opts, const char *name,

                 NetClientState *peer, Error **errp)

{

    /* FIXME error_setg(errp, ...) on failure */

    const NetdevTapOptions *tap;



    assert(opts->kind == NET_CLIENT_OPTIONS_KIND_TAP);

    tap = opts->tap;



    if (!tap->has_ifname) {

        error_report("tap: no interface name");

        return -1;

    }



    if (tap_win32_init(peer, "tap", name, tap->ifname) == -1) {

        return -1;

    }



    return 0;

}
