static void virtio_net_handle_tx_bh(VirtIODevice *vdev, VirtQueue *vq)

{

    VirtIONet *n = to_virtio_net(vdev);



    if (unlikely(n->tx_waiting)) {

        return;

    }

    virtio_queue_set_notification(vq, 0);

    qemu_bh_schedule(n->tx_bh);

    n->tx_waiting = 1;

}
