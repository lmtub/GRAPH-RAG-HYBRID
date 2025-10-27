static void vhost_scsi_realize(DeviceState *dev, Error **errp)

{

    VirtIOSCSICommon *vs = VIRTIO_SCSI_COMMON(dev);

    VHostSCSI *s = VHOST_SCSI(dev);

    Error *err = NULL;

    int vhostfd = -1;

    int ret;



    if (!vs->conf.wwpn) {

        error_setg(errp, "vhost-scsi: missing wwpn");

        return;

    }



    if (vs->conf.vhostfd) {

        vhostfd = monitor_handle_fd_param(cur_mon, vs->conf.vhostfd);

        if (vhostfd == -1) {

            error_setg(errp, "vhost-scsi: unable to parse vhostfd");

            return;

        }

    } else {

        vhostfd = open("/dev/vhost-scsi", O_RDWR);

        if (vhostfd < 0) {

            error_setg(errp, "vhost-scsi: open vhost char device failed: %s",

                       strerror(errno));

            return;

        }

    }



    virtio_scsi_common_realize(dev, &err, vhost_dummy_handle_output,

                               vhost_dummy_handle_output,

                               vhost_dummy_handle_output);

    if (err != NULL) {

        error_propagate(errp, err);


        return;

    }



    s->dev.nvqs = VHOST_SCSI_VQ_NUM_FIXED + vs->conf.num_queues;

    s->dev.vqs = g_new(struct vhost_virtqueue, s->dev.nvqs);

    s->dev.vq_index = 0;

    s->dev.backend_features = 0;



    ret = vhost_dev_init(&s->dev, (void *)(uintptr_t)vhostfd,

                         VHOST_BACKEND_TYPE_KERNEL, true);

    if (ret < 0) {

        error_setg(errp, "vhost-scsi: vhost initialization failed: %s",

                   strerror(-ret));

        return;

    }



    error_setg(&s->migration_blocker,

            "vhost-scsi does not support migration");

    migrate_add_blocker(s->migration_blocker);

}