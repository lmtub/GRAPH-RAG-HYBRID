int net_init_vde(const NetClientOptions *opts, const char *name,

                 NetClientState *peer, Error **errp)

{

    /* FIXME error_setg(errp, ...) on failure */

    const NetdevVdeOptions *vde;



    assert(opts->type == NET_CLIENT_OPTIONS_KIND_VDE);

    vde = opts->u.vde;



    /* missing optional values have been initialized to "all bits zero" */

    if (net_vde_init(peer, "vde", name, vde->sock, vde->port, vde->group,

                     vde->has_mode ? vde->mode : 0700) == -1) {

        return -1;

    }



    return 0;

}
