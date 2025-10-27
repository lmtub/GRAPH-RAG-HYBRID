static void virtio_scsi_device_unrealize(DeviceState *dev, Error **errp)
{
    virtio_scsi_common_unrealize(dev, errp);
}