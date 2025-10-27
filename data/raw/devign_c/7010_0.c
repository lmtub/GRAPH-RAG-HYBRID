SCSIDevice *scsi_bus_legacy_add_drive(SCSIBus *bus, BlockDriverState *bdrv, int unit)

{

    const char *driver;

    DeviceState *dev;



    driver = bdrv_is_sg(bdrv) ? "scsi-generic" : "scsi-disk";

    dev = qdev_create(&bus->qbus, driver);

    qdev_prop_set_uint32(dev, "scsi-id", unit);

    if (qdev_prop_set_drive(dev, "drive", bdrv) < 0) {

        qdev_free(dev);

        return NULL;

    }

    if (qdev_init(dev) < 0)

        return NULL;

    return DO_UPCAST(SCSIDevice, qdev, dev);

}
