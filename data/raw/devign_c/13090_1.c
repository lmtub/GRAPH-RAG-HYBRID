SCSIDevice *scsi_bus_legacy_add_drive(SCSIBus *bus, BlockBackend *blk,

                                      int unit, bool removable, int bootindex,

                                      const char *serial, Error **errp)

{

    const char *driver;

    char *name;

    DeviceState *dev;

    Error *err = NULL;



    driver = blk_is_sg(blk) ? "scsi-generic" : "scsi-disk";

    dev = qdev_create(&bus->qbus, driver);

    name = g_strdup_printf("legacy[%d]", unit);

    object_property_add_child(OBJECT(bus), name, OBJECT(dev), NULL);

    g_free(name);



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

    qdev_prop_set_drive(dev, "drive", blk, &err);

    if (err) {

        qerror_report_err(err);

        error_free(err);

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
