void *qemu_get_virtqueue_element(QEMUFile *f, size_t sz)

{

    VirtQueueElement *elem;

    VirtQueueElementOld data;

    int i;



    qemu_get_buffer(f, (uint8_t *)&data, sizeof(VirtQueueElementOld));



    elem = virtqueue_alloc_element(sz, data.out_num, data.in_num);

    elem->index = data.index;



    for (i = 0; i < elem->in_num; i++) {

        elem->in_addr[i] = data.in_addr[i];

    }



    for (i = 0; i < elem->out_num; i++) {

        elem->out_addr[i] = data.out_addr[i];

    }



    for (i = 0; i < elem->in_num; i++) {

        /* Base is overwritten by virtqueue_map.  */

        elem->in_sg[i].iov_base = 0;

        elem->in_sg[i].iov_len = data.in_sg[i].iov_len;

    }



    for (i = 0; i < elem->out_num; i++) {

        /* Base is overwritten by virtqueue_map.  */

        elem->out_sg[i].iov_base = 0;

        elem->out_sg[i].iov_len = data.out_sg[i].iov_len;

    }



    virtqueue_map(elem);

    return elem;

}
