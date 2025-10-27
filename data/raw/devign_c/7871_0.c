int virtio_set_status(VirtIODevice *vdev, uint8_t val)

{

    VirtioDeviceClass *k = VIRTIO_DEVICE_GET_CLASS(vdev);

    trace_virtio_set_status(vdev, val);



    if (virtio_has_feature(vdev, VIRTIO_F_VERSION_1)) {

        if (!(vdev->status & VIRTIO_CONFIG_S_FEATURES_OK) &&

            val & VIRTIO_CONFIG_S_FEATURES_OK) {

            int ret = virtio_validate_features(vdev);



            if (ret) {

                return ret;

            }

        }

    }

    if (k->set_status) {

        k->set_status(vdev, val);

    }

    vdev->status = val;

    return 0;

}
