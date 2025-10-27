static RxFilterInfo *virtio_net_query_rxfilter(NetClientState *nc)

{

    VirtIONet *n = qemu_get_nic_opaque(nc);

    VirtIODevice *vdev = VIRTIO_DEVICE(n);

    RxFilterInfo *info;

    strList *str_list, *entry;

    int i;



    info = g_malloc0(sizeof(*info));

    info->name = g_strdup(nc->name);

    info->promiscuous = n->promisc;



    if (n->nouni) {

        info->unicast = RX_STATE_NONE;

    } else if (n->alluni) {

        info->unicast = RX_STATE_ALL;

    } else {

        info->unicast = RX_STATE_NORMAL;

    }



    if (n->nomulti) {

        info->multicast = RX_STATE_NONE;

    } else if (n->allmulti) {

        info->multicast = RX_STATE_ALL;

    } else {

        info->multicast = RX_STATE_NORMAL;

    }



    info->broadcast_allowed = n->nobcast;

    info->multicast_overflow = n->mac_table.multi_overflow;

    info->unicast_overflow = n->mac_table.uni_overflow;



    info->main_mac = mac_strdup_printf(n->mac);



    str_list = NULL;

    for (i = 0; i < n->mac_table.first_multi; i++) {

        entry = g_malloc0(sizeof(*entry));

        entry->value = mac_strdup_printf(n->mac_table.macs + i * ETH_ALEN);

        entry->next = str_list;

        str_list = entry;

    }

    info->unicast_table = str_list;



    str_list = NULL;

    for (i = n->mac_table.first_multi; i < n->mac_table.in_use; i++) {

        entry = g_malloc0(sizeof(*entry));

        entry->value = mac_strdup_printf(n->mac_table.macs + i * ETH_ALEN);

        entry->next = str_list;

        str_list = entry;

    }

    info->multicast_table = str_list;

    info->vlan_table = get_vlan_table(n);



    if (!((1 << VIRTIO_NET_F_CTRL_VLAN) & vdev->guest_features)) {

        info->vlan = RX_STATE_ALL;

    } else if (!info->vlan_table) {

        info->vlan = RX_STATE_NONE;

    } else {

        info->vlan = RX_STATE_NORMAL;

    }



    /* enable event notification after query */

    nc->rxfilter_notify_enabled = 1;



    return info;

}
