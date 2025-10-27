static void virtio_net_device_realize(DeviceState *dev, Error **errp)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(dev);

    VirtIONet *n = VIRTIO_NET(dev);

    NetClientState *nc;

    int i;



    virtio_init(vdev, "virtio-net", VIRTIO_ID_NET, n->config_size);



    n->max_queues = MAX(n->nic_conf.peers.queues, 1);








    n->vqs = g_malloc0(sizeof(VirtIONetQueue) * n->max_queues);

    n->vqs[0].rx_vq = virtio_add_queue(vdev, 256, virtio_net_handle_rx);

    n->curr_queues = 1;

    n->vqs[0].n = n;

    n->tx_timeout = n->net_conf.txtimer;



    if (n->net_conf.tx && strcmp(n->net_conf.tx, "timer")

                       && strcmp(n->net_conf.tx, "bh")) {

        error_report("virtio-net: "

                     "Unknown option tx=%s, valid options: \"timer\" \"bh\"",

                     n->net_conf.tx);

        error_report("Defaulting to \"bh\"");




    if (n->net_conf.tx && !strcmp(n->net_conf.tx, "timer")) {

        n->vqs[0].tx_vq = virtio_add_queue(vdev, 256,

                                           virtio_net_handle_tx_timer);

        n->vqs[0].tx_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, virtio_net_tx_timer,

                                               &n->vqs[0]);

    } else {

        n->vqs[0].tx_vq = virtio_add_queue(vdev, 256,

                                           virtio_net_handle_tx_bh);

        n->vqs[0].tx_bh = qemu_bh_new(virtio_net_tx_bh, &n->vqs[0]);


    n->ctrl_vq = virtio_add_queue(vdev, 64, virtio_net_handle_ctrl);

    qemu_macaddr_default_if_unset(&n->nic_conf.macaddr);

    memcpy(&n->mac[0], &n->nic_conf.macaddr, sizeof(n->mac));

    n->status = VIRTIO_NET_S_LINK_UP;

    n->announce_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL,

                                     virtio_net_announce_timer, n);



    if (n->netclient_type) {

        /*

         * Happen when virtio_net_set_netclient_name has been called.

         */

        n->nic = qemu_new_nic(&net_virtio_info, &n->nic_conf,

                              n->netclient_type, n->netclient_name, n);

    } else {

        n->nic = qemu_new_nic(&net_virtio_info, &n->nic_conf,

                              object_get_typename(OBJECT(dev)), dev->id, n);




    peer_test_vnet_hdr(n);

    if (peer_has_vnet_hdr(n)) {

        for (i = 0; i < n->max_queues; i++) {

            qemu_using_vnet_hdr(qemu_get_subqueue(n->nic, i)->peer, true);


        n->host_hdr_len = sizeof(struct virtio_net_hdr);

    } else {

        n->host_hdr_len = 0;




    qemu_format_nic_info_str(qemu_get_queue(n->nic), n->nic_conf.macaddr.a);



    n->vqs[0].tx_waiting = 0;

    n->tx_burst = n->net_conf.txburst;

    virtio_net_set_mrg_rx_bufs(n, 0);

    n->promisc = 1; /* for compatibility */



    n->mac_table.macs = g_malloc0(MAC_TABLE_ENTRIES * ETH_ALEN);



    n->vlans = g_malloc0(MAX_VLAN >> 3);



    nc = qemu_get_queue(n->nic);

    nc->rxfilter_notify_enabled = 1;



    n->qdev = dev;

    register_savevm(dev, "virtio-net", -1, VIRTIO_NET_VM_VERSION,

                    virtio_net_save, virtio_net_load, n);
