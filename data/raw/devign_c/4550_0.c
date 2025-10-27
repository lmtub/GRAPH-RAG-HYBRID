static ssize_t virtio_net_receive(VLANClientState *vc, const uint8_t *buf, size_t size)

{

    VirtIONet *n = vc->opaque;

    struct virtio_net_hdr_mrg_rxbuf *mhdr = NULL;

    size_t hdr_len, offset, i;



    if (!do_virtio_net_can_receive(n, size))

        return 0;



    if (!receive_filter(n, buf, size))

        return size;



    /* hdr_len refers to the header we supply to the guest */

    hdr_len = n->mergeable_rx_bufs ?

        sizeof(struct virtio_net_hdr_mrg_rxbuf) : sizeof(struct virtio_net_hdr);



    offset = i = 0;



    while (offset < size) {

        VirtQueueElement elem;

        int len, total;

        struct iovec sg[VIRTQUEUE_MAX_SIZE];



        len = total = 0;



        if ((i != 0 && !n->mergeable_rx_bufs) ||

            virtqueue_pop(n->rx_vq, &elem) == 0) {

            if (i == 0)

                return -1;

            fprintf(stderr, "virtio-net truncating packet\n");

            exit(1);

        }



        if (elem.in_num < 1) {

            fprintf(stderr, "virtio-net receive queue contains no in buffers\n");

            exit(1);

        }



        if (!n->mergeable_rx_bufs && elem.in_sg[0].iov_len != hdr_len) {

            fprintf(stderr, "virtio-net header not in first element\n");

            exit(1);

        }



        memcpy(&sg, &elem.in_sg[0], sizeof(sg[0]) * elem.in_num);



        if (i == 0) {

            if (n->mergeable_rx_bufs)

                mhdr = (struct virtio_net_hdr_mrg_rxbuf *)sg[0].iov_base;



            offset += receive_header(n, sg, elem.in_num,

                                     buf + offset, size - offset, hdr_len);

            total += hdr_len;

        }



        /* copy in packet.  ugh */

        len = iov_fill(sg, elem.in_num,

                       buf + offset, size - offset);

        total += len;



        /* signal other side */

        virtqueue_fill(n->rx_vq, &elem, total, i++);



        offset += len;

    }



    if (mhdr)

        mhdr->num_buffers = i;



    virtqueue_flush(n->rx_vq, i);

    virtio_notify(&n->vdev, n->rx_vq);



    return size;

}
