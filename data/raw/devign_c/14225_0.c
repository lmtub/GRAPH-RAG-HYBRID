int net_init_hubport(const NetClientOptions *opts, const char *name,

                     NetClientState *peer, Error **errp)

{

    const NetdevHubPortOptions *hubport;



    assert(opts->type == NET_CLIENT_OPTIONS_KIND_HUBPORT);

    assert(!peer);

    hubport = opts->u.hubport;



    net_hub_add_port(hubport->hubid, name);

    return 0;

}
