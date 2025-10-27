void virtqueue_fill(VirtQueue *vq, const VirtQueueElement *elem,
                    unsigned int len, unsigned int idx)
{
    VRingUsedElem uelem;
    trace_virtqueue_fill(vq, elem, len, idx);
    virtqueue_unmap_sg(vq, elem, len);
    idx = (idx + vq->used_idx) % vq->vring.num;
    uelem.id = elem->index;
    uelem.len = len;
    vring_used_write(vq, &uelem, idx);