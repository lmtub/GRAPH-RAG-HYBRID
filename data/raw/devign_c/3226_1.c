void virtio_scsi_push_event(VirtIOSCSI *s, SCSIDevice *dev,

                            uint32_t event, uint32_t reason)

{

    VirtIOSCSICommon *vs = VIRTIO_SCSI_COMMON(s);

    VirtIOSCSIReq *req;

    VirtIOSCSIEvent *evt;

    VirtIODevice *vdev = VIRTIO_DEVICE(s);



    if (!(vdev->status & VIRTIO_CONFIG_S_DRIVER_OK)) {

        return;

    }



    if (s->dataplane_started) {

        assert(s->ctx);

        aio_context_acquire(s->ctx);

    }



    req = virtio_scsi_pop_req(s, vs->event_vq);

    if (!req) {

        s->events_dropped = true;

        goto out;

    }



    if (s->events_dropped) {

        event |= VIRTIO_SCSI_T_EVENTS_MISSED;

        s->events_dropped = false;

    }



    if (virtio_scsi_parse_req(req, 0, sizeof(VirtIOSCSIEvent))) {

        virtio_scsi_bad_req();

    }



    evt = &req->resp.event;

    memset(evt, 0, sizeof(VirtIOSCSIEvent));

    evt->event = virtio_tswap32(vdev, event);

    evt->reason = virtio_tswap32(vdev, reason);

    if (!dev) {

        assert(event == VIRTIO_SCSI_T_EVENTS_MISSED);

    } else {

        evt->lun[0] = 1;

        evt->lun[1] = dev->id;



        /* Linux wants us to keep the same encoding we use for REPORT LUNS.  */

        if (dev->lun >= 256) {

            evt->lun[2] = (dev->lun >> 8) | 0x40;

        }

        evt->lun[3] = dev->lun & 0xFF;

    }

    virtio_scsi_complete_req(req);

out:

    if (s->dataplane_started) {

        aio_context_release(s->ctx);

    }

}
