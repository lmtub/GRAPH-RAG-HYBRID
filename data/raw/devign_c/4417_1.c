static void virtio_net_handle_ctrl(VirtIODevice *vdev, VirtQueue *vq)

{

    VirtIONet *n = VIRTIO_NET(vdev);

    struct virtio_net_ctrl_hdr ctrl;

    virtio_net_ctrl_ack status = VIRTIO_NET_ERR;

    VirtQueueElement elem;

    size_t s;

    struct iovec *iov;

    unsigned int iov_cnt;



    while (virtqueue_pop(vq, &elem)) {

        if (iov_size(elem.in_sg, elem.in_num) < sizeof(status) ||

            iov_size(elem.out_sg, elem.out_num) < sizeof(ctrl)) {

            error_report("virtio-net ctrl missing headers");

            exit(1);

        }



        iov = elem.out_sg;

        iov_cnt = elem.out_num;

        s = iov_to_buf(iov, iov_cnt, 0, &ctrl, sizeof(ctrl));

        iov_discard_front(&iov, &iov_cnt, sizeof(ctrl));

        if (s != sizeof(ctrl)) {

            status = VIRTIO_NET_ERR;

        } else if (ctrl.class == VIRTIO_NET_CTRL_RX) {

            status = virtio_net_handle_rx_mode(n, ctrl.cmd, iov, iov_cnt);

        } else if (ctrl.class == VIRTIO_NET_CTRL_MAC) {

            status = virtio_net_handle_mac(n, ctrl.cmd, iov, iov_cnt);

        } else if (ctrl.class == VIRTIO_NET_CTRL_VLAN) {

            status = virtio_net_handle_vlan_table(n, ctrl.cmd, iov, iov_cnt);

        } else if (ctrl.class == VIRTIO_NET_CTRL_ANNOUNCE) {

            status = virtio_net_handle_announce(n, ctrl.cmd, iov, iov_cnt);

        } else if (ctrl.class == VIRTIO_NET_CTRL_MQ) {

            status = virtio_net_handle_mq(n, ctrl.cmd, iov, iov_cnt);

        } else if (ctrl.class == VIRTIO_NET_CTRL_GUEST_OFFLOADS) {

            status = virtio_net_handle_offloads(n, ctrl.cmd, iov, iov_cnt);

        }



        s = iov_from_buf(elem.in_sg, elem.in_num, 0, &status, sizeof(status));

        assert(s == sizeof(status));



        virtqueue_push(vq, &elem, sizeof(status));

        virtio_notify(vdev, vq);

    }

}
