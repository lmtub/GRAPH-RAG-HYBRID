void qdev_prop_set_drive_nofail(DeviceState *dev, const char *name,

                                BlockDriverState *value)

{

    if (qdev_prop_set_drive(dev, name, value) < 0) {

        exit(1);

    }

}
