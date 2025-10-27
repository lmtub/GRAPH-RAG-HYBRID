vu_queue_fill(VuDev *dev, VuVirtq *vq,

              const VuVirtqElement *elem,

              unsigned int len, unsigned int idx)

{

    struct vring_used_elem uelem;



    if (unlikely(dev->broken)) {

        return;

    }



    vu_log_queue_fill(dev, vq, elem, len);



    idx = (idx + vq->used_idx) % vq->vring.num;



    uelem.id = elem->index;

    uelem.len = len;

    vring_used_write(dev, vq, &uelem, idx);

}
