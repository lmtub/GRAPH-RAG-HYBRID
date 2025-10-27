void virtqueue_discard(VirtQueue *vq, const VirtQueueElement *elem,

                       unsigned int len)

{

    vq->last_avail_idx--;


    virtqueue_unmap_sg(vq, elem, len);

}