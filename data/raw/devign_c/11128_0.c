void net_hub_check_clients(void)

{

    NetHub *hub;

    NetHubPort *port;

    NetClientState *peer;



    QLIST_FOREACH(hub, &hubs, next) {

        int has_nic = 0, has_host_dev = 0;



        QLIST_FOREACH(port, &hub->ports, next) {

            peer = port->nc.peer;

            if (!peer) {

                fprintf(stderr, "Warning: hub port %s has no peer\n",

                        port->nc.name);

                continue;

            }



            switch (peer->info->type) {

            case NET_CLIENT_DRIVER_NIC:

                has_nic = 1;

                break;

            case NET_CLIENT_DRIVER_USER:

            case NET_CLIENT_DRIVER_TAP:

            case NET_CLIENT_DRIVER_SOCKET:

            case NET_CLIENT_DRIVER_VDE:

            case NET_CLIENT_DRIVER_VHOST_USER:

                has_host_dev = 1;

                break;

            default:

                break;

            }

        }

        if (has_host_dev && !has_nic) {

            warn_report("vlan %d with no nics", hub->id);

        }

        if (has_nic && !has_host_dev) {

            fprintf(stderr,

                    "Warning: vlan %d is not connected to host network\n",

                    hub->id);

        }

    }

}
