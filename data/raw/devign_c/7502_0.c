static void virtio_net_tx_complete(NetClientState *nc, ssize_t len)

{

    VirtIONet *n = qemu_get_nic_opaque(nc);

    VirtIONetQueue *q = virtio_net_get_subqueue(nc);

    VirtIODevice *vdev = VIRTIO_DEVICE(n);



    virtqueue_push(q->tx_vq, &q->async_tx.elem, 0);

    virtio_notify(vdev, q->tx_vq);



    q->async_tx.elem.out_num = 0;



    virtio_queue_set_notification(q->tx_vq, 1);

    virtio_net_flush_tx(q);

}
