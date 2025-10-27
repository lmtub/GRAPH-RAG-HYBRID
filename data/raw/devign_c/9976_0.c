static int net_client_init1(const void *object, int is_netdev, Error **errp)

{

    union {

        const Netdev    *netdev;

        const NetLegacy *net;

    } u;

    const NetClientOptions *opts;

    const char *name;



    if (is_netdev) {

        u.netdev = object;

        opts = u.netdev->opts;

        name = u.netdev->id;



        if (opts->kind == NET_CLIENT_OPTIONS_KIND_DUMP ||

            opts->kind == NET_CLIENT_OPTIONS_KIND_NIC ||

            !net_client_init_fun[opts->kind]) {

            error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "type",

                       "a netdev backend type");

            return -1;

        }

    } else {

        u.net = object;

        opts = u.net->opts;

        if (opts->kind == NET_CLIENT_OPTIONS_KIND_HUBPORT) {

            error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "type",

                       "a net type");

            return -1;

        }

        /* missing optional values have been initialized to "all bits zero" */

        name = u.net->has_id ? u.net->id : u.net->name;



        if (opts->kind == NET_CLIENT_OPTIONS_KIND_NONE) {

            return 0; /* nothing to do */

        }



        if (!net_client_init_fun[opts->kind]) {

            error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "type",

                       "a net backend type (maybe it is not compiled "

                       "into this binary)");

            return -1;

        }

    }



    if (net_client_init_fun[opts->kind]) {

        NetClientState *peer = NULL;



        /* Do not add to a vlan if it's a -netdev or a nic with a netdev=

         * parameter. */

        if (!is_netdev &&

            (opts->kind != NET_CLIENT_OPTIONS_KIND_NIC ||

             !opts->nic->has_netdev)) {

            peer = net_hub_add_port(u.net->has_vlan ? u.net->vlan : 0, NULL);

        }



        if (net_client_init_fun[opts->kind](opts, name, peer, errp) < 0) {

            /* FIXME drop when all init functions store an Error */

            if (errp && !*errp) {

                error_setg(errp, QERR_DEVICE_INIT_FAILED,

                           NetClientOptionsKind_lookup[opts->kind]);

            }

            return -1;

        }

    }

    return 0;

}
