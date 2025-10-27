static void virtio_scsi_vring_teardown(VirtIOSCSI *s)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(s);

    VirtIOSCSICommon *vs = VIRTIO_SCSI_COMMON(s);

    int i;



    if (s->ctrl_vring) {

        vring_teardown(&s->ctrl_vring->vring, vdev, 0);



    }

    if (s->event_vring) {

        vring_teardown(&s->event_vring->vring, vdev, 1);



    }

    if (s->cmd_vrings) {

        for (i = 0; i < vs->conf.num_queues && s->cmd_vrings[i]; i++) {

            vring_teardown(&s->cmd_vrings[i]->vring, vdev, 2 + i);



        }

        free(s->cmd_vrings);

        s->cmd_vrings = NULL;

    }

}