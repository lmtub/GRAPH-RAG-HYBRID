void GCC_FMT_ATTR(2, 3) virtio_error(VirtIODevice *vdev, const char *fmt, ...)

{

    va_list ap;



    va_start(ap, fmt);

    error_vreport(fmt, ap);

    va_end(ap);



    vdev->broken = true;



    if (virtio_vdev_has_feature(vdev, VIRTIO_F_VERSION_1)) {

        virtio_set_status(vdev, vdev->status | VIRTIO_CONFIG_S_NEEDS_RESET);

        virtio_notify_config(vdev);

    }

}
