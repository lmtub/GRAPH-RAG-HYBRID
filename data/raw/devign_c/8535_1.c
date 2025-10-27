GuestNetworkInterfaceList *qmp_guest_network_get_interfaces(Error **errp)

{

    GuestNetworkInterfaceList *head = NULL, *cur_item = NULL;

    struct ifaddrs *ifap, *ifa;



    if (getifaddrs(&ifap) < 0) {

        error_setg_errno(errp, errno, "getifaddrs failed");

        goto error;

    }



    for (ifa = ifap; ifa; ifa = ifa->ifa_next) {

        GuestNetworkInterfaceList *info;

        GuestIpAddressList **address_list = NULL, *address_item = NULL;

        char addr4[INET_ADDRSTRLEN];

        char addr6[INET6_ADDRSTRLEN];

        int sock;

        struct ifreq ifr;

        unsigned char *mac_addr;

        void *p;



        g_debug("Processing %s interface", ifa->ifa_name);



        info = guest_find_interface(head, ifa->ifa_name);



        if (!info) {

            info = g_malloc0(sizeof(*info));

            info->value = g_malloc0(sizeof(*info->value));

            info->value->name = g_strdup(ifa->ifa_name);



            if (!cur_item) {

                head = cur_item = info;

            } else {

                cur_item->next = info;

                cur_item = info;

            }

        }



        if (!info->value->has_hardware_address &&

            ifa->ifa_flags & SIOCGIFHWADDR) {

            /* we haven't obtained HW address yet */

            sock = socket(PF_INET, SOCK_STREAM, 0);

            if (sock == -1) {

                error_setg_errno(errp, errno, "failed to create socket");

                goto error;

            }



            memset(&ifr, 0, sizeof(ifr));

            pstrcpy(ifr.ifr_name, IF_NAMESIZE, info->value->name);

            if (ioctl(sock, SIOCGIFHWADDR, &ifr) == -1) {

                error_setg_errno(errp, errno,

                                 "failed to get MAC address of %s",

                                 ifa->ifa_name);

                goto error;

            }



            mac_addr = (unsigned char *) &ifr.ifr_hwaddr.sa_data;



            info->value->hardware_address =

                g_strdup_printf("%02x:%02x:%02x:%02x:%02x:%02x",

                                (int) mac_addr[0], (int) mac_addr[1],

                                (int) mac_addr[2], (int) mac_addr[3],

                                (int) mac_addr[4], (int) mac_addr[5]);



            info->value->has_hardware_address = true;

            close(sock);

        }



        if (ifa->ifa_addr &&

            ifa->ifa_addr->sa_family == AF_INET) {

            /* interface with IPv4 address */

            address_item = g_malloc0(sizeof(*address_item));

            address_item->value = g_malloc0(sizeof(*address_item->value));

            p = &((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;

            if (!inet_ntop(AF_INET, p, addr4, sizeof(addr4))) {

                error_setg_errno(errp, errno, "inet_ntop failed");

                goto error;

            }



            address_item->value->ip_address = g_strdup(addr4);

            address_item->value->ip_address_type = GUEST_IP_ADDRESS_TYPE_IPV4;



            if (ifa->ifa_netmask) {

                /* Count the number of set bits in netmask.

                 * This is safe as '1' and '0' cannot be shuffled in netmask. */

                p = &((struct sockaddr_in *)ifa->ifa_netmask)->sin_addr;

                address_item->value->prefix = ctpop32(((uint32_t *) p)[0]);

            }

        } else if (ifa->ifa_addr &&

                   ifa->ifa_addr->sa_family == AF_INET6) {

            /* interface with IPv6 address */

            address_item = g_malloc0(sizeof(*address_item));

            address_item->value = g_malloc0(sizeof(*address_item->value));

            p = &((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_addr;

            if (!inet_ntop(AF_INET6, p, addr6, sizeof(addr6))) {

                error_setg_errno(errp, errno, "inet_ntop failed");

                goto error;

            }



            address_item->value->ip_address = g_strdup(addr6);

            address_item->value->ip_address_type = GUEST_IP_ADDRESS_TYPE_IPV6;



            if (ifa->ifa_netmask) {

                /* Count the number of set bits in netmask.

                 * This is safe as '1' and '0' cannot be shuffled in netmask. */

                p = &((struct sockaddr_in6 *)ifa->ifa_netmask)->sin6_addr;

                address_item->value->prefix =

                    ctpop32(((uint32_t *) p)[0]) +

                    ctpop32(((uint32_t *) p)[1]) +

                    ctpop32(((uint32_t *) p)[2]) +

                    ctpop32(((uint32_t *) p)[3]);

            }

        }



        if (!address_item) {

            continue;

        }



        address_list = &info->value->ip_addresses;



        while (*address_list && (*address_list)->next) {

            address_list = &(*address_list)->next;

        }



        if (!*address_list) {

            *address_list = address_item;

        } else {

            (*address_list)->next = address_item;

        }



        info->value->has_ip_addresses = true;





    }



    freeifaddrs(ifap);

    return head;



error:

    freeifaddrs(ifap);

    qapi_free_GuestNetworkInterfaceList(head);

    return NULL;

}
