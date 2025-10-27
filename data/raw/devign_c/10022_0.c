static void handle_notify(EventNotifier *e)

{

    VirtIOBlockDataPlane *s = container_of(e, VirtIOBlockDataPlane,

                                           host_notifier);

    VirtIOBlock *vblk = VIRTIO_BLK(s->vdev);



    event_notifier_test_and_clear(&s->host_notifier);

    blk_io_plug(s->conf->conf.blk);

    for (;;) {

        MultiReqBuffer mrb = {};

        int ret;



        /* Disable guest->host notifies to avoid unnecessary vmexits */

        vring_disable_notification(s->vdev, &s->vring);



        for (;;) {

            VirtIOBlockReq *req = virtio_blk_alloc_request(vblk);



            ret = vring_pop(s->vdev, &s->vring, &req->elem);

            if (ret < 0) {

                virtio_blk_free_request(req);

                break; /* no more requests */

            }



            trace_virtio_blk_data_plane_process_request(s, req->elem.out_num,

                                                        req->elem.in_num,

                                                        req->elem.index);



            virtio_blk_handle_request(req, &mrb);

        }



        if (mrb.num_reqs) {

            virtio_blk_submit_multireq(s->conf->conf.blk, &mrb);

        }



        if (likely(ret == -EAGAIN)) { /* vring emptied */

            /* Re-enable guest->host notifies and stop processing the vring.

             * But if the guest has snuck in more descriptors, keep processing.

             */

            if (vring_enable_notification(s->vdev, &s->vring)) {

                break;

            }

        } else { /* fatal error */

            break;

        }

    }

    blk_io_unplug(s->conf->conf.blk);

}
