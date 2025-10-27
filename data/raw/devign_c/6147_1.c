vu_queue_pop(VuDev *dev, VuVirtq *vq, size_t sz)

{

    unsigned int i, head, max;

    VuVirtqElement *elem;

    unsigned out_num, in_num;

    struct iovec iov[VIRTQUEUE_MAX_SIZE];

    struct vring_desc *desc;

    int rc;



    if (unlikely(dev->broken)) {

        return NULL;

    }



    if (vu_queue_empty(dev, vq)) {

        return NULL;

    }

    /* Needed after virtio_queue_empty(), see comment in

     * virtqueue_num_heads(). */

    smp_rmb();



    /* When we start there are none of either input nor output. */

    out_num = in_num = 0;



    max = vq->vring.num;

    if (vq->inuse >= vq->vring.num) {

        vu_panic(dev, "Virtqueue size exceeded");

        return NULL;

    }



    if (!virtqueue_get_head(dev, vq, vq->last_avail_idx++, &head)) {

        return NULL;

    }



    if (vu_has_feature(dev, VIRTIO_RING_F_EVENT_IDX)) {

        vring_set_avail_event(vq, vq->last_avail_idx);

    }



    i = head;

    desc = vq->vring.desc;

    if (desc[i].flags & VRING_DESC_F_INDIRECT) {

        if (desc[i].len % sizeof(struct vring_desc)) {

            vu_panic(dev, "Invalid size for indirect buffer table");

        }



        /* loop over the indirect descriptor table */

        max = desc[i].len / sizeof(struct vring_desc);

        desc = vu_gpa_to_va(dev, desc[i].addr);

        i = 0;

    }



    /* Collect all the descriptors */

    do {

        if (desc[i].flags & VRING_DESC_F_WRITE) {

            virtqueue_map_desc(dev, &in_num, iov + out_num,

                               VIRTQUEUE_MAX_SIZE - out_num, true,

                               desc[i].addr, desc[i].len);

        } else {

            if (in_num) {

                vu_panic(dev, "Incorrect order for descriptors");

                return NULL;

            }

            virtqueue_map_desc(dev, &out_num, iov,

                               VIRTQUEUE_MAX_SIZE, false,

                               desc[i].addr, desc[i].len);

        }



        /* If we've got too many, that implies a descriptor loop. */

        if ((in_num + out_num) > max) {

            vu_panic(dev, "Looped descriptor");

        }

        rc = virtqueue_read_next_desc(dev, desc, i, max, &i);

    } while (rc == VIRTQUEUE_READ_DESC_MORE);



    if (rc == VIRTQUEUE_READ_DESC_ERROR) {

        return NULL;

    }



    /* Now copy what we have collected and mapped */

    elem = virtqueue_alloc_element(sz, out_num, in_num);

    elem->index = head;

    for (i = 0; i < out_num; i++) {

        elem->out_sg[i] = iov[i];

    }

    for (i = 0; i < in_num; i++) {

        elem->in_sg[i] = iov[out_num + i];

    }



    vq->inuse++;



    return elem;

}
