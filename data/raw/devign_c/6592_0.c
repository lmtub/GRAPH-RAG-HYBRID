static int virtio_blk_load_device(VirtIODevice *vdev, QEMUFile *f,

                                  int version_id)

{

    VirtIOBlock *s = VIRTIO_BLK(vdev);



    while (qemu_get_sbyte(f)) {

        unsigned nvqs = s->conf.num_queues;

        unsigned vq_idx = 0;

        VirtIOBlockReq *req;



        if (nvqs > 1) {

            vq_idx = qemu_get_be32(f);



            if (vq_idx >= nvqs) {

                error_report("Invalid virtqueue index in request list: %#x",

                             vq_idx);

                return -EINVAL;

            }

        }



        req = qemu_get_virtqueue_element(f, sizeof(VirtIOBlockReq));

        virtio_blk_init_request(s, virtio_get_queue(vdev, vq_idx), req);

        req->next = s->rq;

        s->rq = req;

    }



    return 0;

}
