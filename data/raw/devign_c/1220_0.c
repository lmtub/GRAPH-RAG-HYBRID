static void virtio_net_set_features(VirtIODevice *vdev, uint32_t features)

{

    VirtIONet *n = VIRTIO_NET(vdev);

    int i;



    virtio_net_set_multiqueue(n, !!(features & (1 << VIRTIO_NET_F_MQ)),

                              !!(features & (1 << VIRTIO_NET_F_CTRL_VQ)));



    virtio_net_set_mrg_rx_bufs(n, !!(features & (1 << VIRTIO_NET_F_MRG_RXBUF)));



    if (n->has_vnet_hdr) {

        tap_set_offload(qemu_get_subqueue(n->nic, 0)->peer,

                        (features >> VIRTIO_NET_F_GUEST_CSUM) & 1,

                        (features >> VIRTIO_NET_F_GUEST_TSO4) & 1,

                        (features >> VIRTIO_NET_F_GUEST_TSO6) & 1,

                        (features >> VIRTIO_NET_F_GUEST_ECN)  & 1,

                        (features >> VIRTIO_NET_F_GUEST_UFO)  & 1);

    }



    for (i = 0;  i < n->max_queues; i++) {

        NetClientState *nc = qemu_get_subqueue(n->nic, i);



        if (!nc->peer || nc->peer->info->type != NET_CLIENT_OPTIONS_KIND_TAP) {

            continue;

        }

        if (!tap_get_vhost_net(nc->peer)) {

            continue;

        }

        vhost_net_ack_features(tap_get_vhost_net(nc->peer), features);

    }

}
