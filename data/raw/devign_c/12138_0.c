VirtIODevice *virtio_net_init(DeviceState *dev)

{

    VirtIONet *n;

    static int virtio_net_id;



    n = (VirtIONet *)virtio_common_init("virtio-net", VIRTIO_ID_NET,

                                        sizeof(struct virtio_net_config),

                                        sizeof(VirtIONet));



    n->vdev.get_config = virtio_net_get_config;

    n->vdev.set_config = virtio_net_set_config;

    n->vdev.get_features = virtio_net_get_features;

    n->vdev.set_features = virtio_net_set_features;

    n->vdev.bad_features = virtio_net_bad_features;

    n->vdev.reset = virtio_net_reset;

    n->rx_vq = virtio_add_queue(&n->vdev, 256, virtio_net_handle_rx);

    n->tx_vq = virtio_add_queue(&n->vdev, 256, virtio_net_handle_tx);

    n->ctrl_vq = virtio_add_queue(&n->vdev, 16, virtio_net_handle_ctrl);

    qdev_get_macaddr(dev, n->mac);

    n->status = VIRTIO_NET_S_LINK_UP;

    n->vc = qdev_get_vlan_client(dev,

                                 virtio_net_can_receive,

                                 virtio_net_receive, NULL,

                                 virtio_net_cleanup, n);

    n->vc->link_status_changed = virtio_net_set_link_status;



    qemu_format_nic_info_str(n->vc, n->mac);



    n->tx_timer = qemu_new_timer(vm_clock, virtio_net_tx_timer, n);

    n->tx_timer_active = 0;

    n->mergeable_rx_bufs = 0;

    n->promisc = 1; /* for compatibility */



    n->mac_table.macs = qemu_mallocz(MAC_TABLE_ENTRIES * ETH_ALEN);



    n->vlans = qemu_mallocz(MAX_VLAN >> 3);



    register_savevm("virtio-net", virtio_net_id++, VIRTIO_NET_VM_VERSION,

                    virtio_net_save, virtio_net_load, n);



    return &n->vdev;

}
