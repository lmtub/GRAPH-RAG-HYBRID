void virtqueue_map(VirtIODevice *vdev, VirtQueueElement *elem)

{

    virtqueue_map_iovec(vdev, elem->in_sg, elem->in_addr, &elem->in_num,

                        MIN(ARRAY_SIZE(elem->in_sg), ARRAY_SIZE(elem->in_addr)),

                        1);

    virtqueue_map_iovec(vdev, elem->out_sg, elem->out_addr, &elem->out_num,

                        MIN(ARRAY_SIZE(elem->out_sg),

                        ARRAY_SIZE(elem->out_addr)),

                        0);

}
