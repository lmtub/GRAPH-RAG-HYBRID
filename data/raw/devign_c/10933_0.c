SCSIDevice *scsi_bus_legacy_add_drive(SCSIBus *bus, BlockDriverState *bdrv,

                                      int unit, bool removable, int bootindex,

                                      const char *serial, Error **errp)

{

    const char *driver;

    DeviceState *dev;

    Error *err = NULL;



    driver = bdrv_is_sg(bdrv) ? "scsi-generic" : "scsi-disk";

    dev = qdev_create(&bus->qbus, driver);

    qdev_prop_set_uint32(dev, "scsi-id", unit);

    if (bootindex >= 0) {

        object_property_set_int(OBJECT(dev), bootindex, "bootindex",

                                &error_abort);

    }

    if (object_property_find(OBJECT(dev), "removable", NULL)) {

        qdev_prop_set_bit(dev, "removable", removable);

    }

    if (serial && object_property_find(OBJECT(dev), "serial", NULL)) {

        qdev_prop_set_string(dev, "serial", serial);

    }

    if (qdev_prop_set_drive(dev, "drive", bdrv) < 0) {

        error_setg(errp, "Setting drive property failed");

        object_unparent(OBJECT(dev));

        return NULL;

    }

    object_property_set_bool(OBJECT(dev), true, "realized", &err);

    if (err != NULL) {

        error_propagate(errp, err);

        object_unparent(OBJECT(dev));

        return NULL;

    }

    return SCSI_DEVICE(dev);

}
