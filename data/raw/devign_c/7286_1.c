static int virtio_blk_load(QEMUFile *f, void *opaque, int version_id)

{

    VirtIOBlock *s = opaque;



    if (version_id != 2)

        return -EINVAL;



    virtio_load(&s->vdev, f);

    while (qemu_get_sbyte(f)) {

        VirtIOBlockReq *req = virtio_blk_alloc_request(s);

        qemu_get_buffer(f, (unsigned char*)&req->elem, sizeof(req->elem));

        req->next = s->rq;

        s->rq = req;



        virtqueue_map_sg(req->elem.in_sg, req->elem.in_addr,

            req->elem.in_num, 1);

        virtqueue_map_sg(req->elem.out_sg, req->elem.out_addr,

            req->elem.out_num, 0);

    }



    return 0;

}
