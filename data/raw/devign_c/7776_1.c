bool virtio_scsi_handle_cmd_vq(VirtIOSCSI *s, VirtQueue *vq)

{

    VirtIOSCSIReq *req, *next;

    int ret = 0;

    bool progress = false;



    QTAILQ_HEAD(, VirtIOSCSIReq) reqs = QTAILQ_HEAD_INITIALIZER(reqs);



    virtio_scsi_acquire(s);

    do {

        virtio_queue_set_notification(vq, 0);



        while ((req = virtio_scsi_pop_req(s, vq))) {

            progress = true;

            ret = virtio_scsi_handle_cmd_req_prepare(s, req);

            if (!ret) {

                QTAILQ_INSERT_TAIL(&reqs, req, next);

            } else if (ret == -EINVAL) {

                /* The device is broken and shouldn't process any request */

                while (!QTAILQ_EMPTY(&reqs)) {

                    req = QTAILQ_FIRST(&reqs);

                    QTAILQ_REMOVE(&reqs, req, next);

                    blk_io_unplug(req->sreq->dev->conf.blk);

                    scsi_req_unref(req->sreq);

                    virtqueue_detach_element(req->vq, &req->elem, 0);

                    virtio_scsi_free_req(req);

                }

            }

        }



        virtio_queue_set_notification(vq, 1);

    } while (ret != -EINVAL && !virtio_queue_empty(vq));



    QTAILQ_FOREACH_SAFE(req, &reqs, next, next) {

        virtio_scsi_handle_cmd_req_submit(s, req);

    }

    virtio_scsi_release(s);

    return progress;

}
