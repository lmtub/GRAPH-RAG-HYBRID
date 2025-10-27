void virtqueue_map(VirtQueueElement *elem)

{

    virtqueue_map_iovec(elem->in_sg, elem->in_addr, &elem->in_num,

                        VIRTQUEUE_MAX_SIZE, 1);

    virtqueue_map_iovec(elem->out_sg, elem->out_addr, &elem->out_num,

                        VIRTQUEUE_MAX_SIZE, 0);

}
