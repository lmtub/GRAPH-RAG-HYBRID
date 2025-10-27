void *qemu_get_virtqueue_element(VirtIODevice *vdev, QEMUFile *f, size_t sz)

{

    VirtQueueElement *elem;

    VirtQueueElementOld data;

    int i;



    qemu_get_buffer(f, (uint8_t *)&data, sizeof(VirtQueueElementOld));



    /* TODO: teach all callers that this can fail, and return failure instead

     * of asserting here.

     * When we do, we might be able to re-enable NDEBUG below.

     */

#ifdef NDEBUG

#error building with NDEBUG is not supported

#endif

    assert(ARRAY_SIZE(data.in_addr) >= data.in_num);

    assert(ARRAY_SIZE(data.out_addr) >= data.out_num);



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



    virtqueue_map(vdev, elem);

    return elem;

}
