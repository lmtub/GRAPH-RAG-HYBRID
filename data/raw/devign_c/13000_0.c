static void virtio_net_set_features(VirtIODevice *vdev, uint64_t features)

{

    VirtIONet *n = VIRTIO_NET(vdev);

    int i;



    virtio_net_set_multiqueue(n,

                              __virtio_has_feature(features, VIRTIO_NET_F_MQ));



    virtio_net_set_mrg_rx_bufs(n,

                               __virtio_has_feature(features,

                                                    VIRTIO_NET_F_MRG_RXBUF),

                               __virtio_has_feature(features,

                                                    VIRTIO_F_VERSION_1));



    if (n->has_vnet_hdr) {

        n->curr_guest_offloads =

            virtio_net_guest_offloads_by_features(features);

        virtio_net_apply_guest_offloads(n);

    }



    for (i = 0;  i < n->max_queues; i++) {

        NetClientState *nc = qemu_get_subqueue(n->nic, i);



        if (!get_vhost_net(nc->peer)) {

            continue;

        }

        vhost_net_ack_features(get_vhost_net(nc->peer), features);

    }



    if (__virtio_has_feature(features, VIRTIO_NET_F_CTRL_VLAN)) {

        memset(n->vlans, 0, MAX_VLAN >> 3);

    } else {

        memset(n->vlans, 0xff, MAX_VLAN >> 3);

    }

}
