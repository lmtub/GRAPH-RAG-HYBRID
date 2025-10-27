void *virtqueue_alloc_element(size_t sz, unsigned out_num, unsigned in_num)

{

    VirtQueueElement *elem;

    size_t in_addr_ofs = QEMU_ALIGN_UP(sz, __alignof__(elem->in_addr[0]));

    size_t out_addr_ofs = in_addr_ofs + in_num * sizeof(elem->in_addr[0]);

    size_t out_addr_end = out_addr_ofs + out_num * sizeof(elem->out_addr[0]);

    size_t in_sg_ofs = QEMU_ALIGN_UP(out_addr_end, __alignof__(elem->in_sg[0]));

    size_t out_sg_ofs = in_sg_ofs + in_num * sizeof(elem->in_sg[0]);

    size_t out_sg_end = out_sg_ofs + out_num * sizeof(elem->out_sg[0]);



    assert(sz >= sizeof(VirtQueueElement));

    elem = g_malloc(out_sg_end);

    elem->out_num = out_num;

    elem->in_num = in_num;

    elem->in_addr = (void *)elem + in_addr_ofs;

    elem->out_addr = (void *)elem + out_addr_ofs;

    elem->in_sg = (void *)elem + in_sg_ofs;

    elem->out_sg = (void *)elem + out_sg_ofs;

    return elem;

}
