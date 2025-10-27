static void vmxnet3_net_init(VMXNET3State *s)

{

    DeviceState *d = DEVICE(s);



    VMW_CBPRN("vmxnet3_net_init called...");



    qemu_macaddr_default_if_unset(&s->conf.macaddr);



    /* Windows guest will query the address that was set on init */

    memcpy(&s->perm_mac.a, &s->conf.macaddr.a, sizeof(s->perm_mac.a));



    s->mcast_list = NULL;

    s->mcast_list_len = 0;



    s->link_status_and_speed = VMXNET3_LINK_SPEED | VMXNET3_LINK_STATUS_UP;



    VMW_CFPRN("Permanent MAC: " MAC_FMT, MAC_ARG(s->perm_mac.a));



    s->nic = qemu_new_nic(&net_vmxnet3_info, &s->conf,

                          object_get_typename(OBJECT(s)),

                          d->id, s);



    s->peer_has_vhdr = vmxnet3_peer_has_vnet_hdr(s);

    s->tx_sop = true;

    s->skip_current_tx_pkt = false;

    s->tx_pkt = NULL;

    s->rx_pkt = NULL;

    s->rx_vlan_stripping = false;

    s->lro_supported = false;



    if (s->peer_has_vhdr) {

        qemu_peer_set_vnet_hdr_len(qemu_get_queue(s->nic),

            sizeof(struct virtio_net_hdr));



        qemu_peer_using_vnet_hdr(qemu_get_queue(s->nic), 1);

    }



    qemu_format_nic_info_str(qemu_get_queue(s->nic), s->conf.macaddr.a);

}
