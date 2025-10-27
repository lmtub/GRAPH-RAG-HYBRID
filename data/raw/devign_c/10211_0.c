int qdev_prop_set_drive(DeviceState *dev, const char *name, BlockDriverState *value)

{

    int res;



    res = bdrv_attach(value, dev);

    if (res < 0) {

        error_report("Can't attach drive %s to %s.%s: %s",

                     bdrv_get_device_name(value),

                     dev->id ? dev->id : dev->info->name,

                     name, strerror(-res));

        return -1;

    }

    qdev_prop_set(dev, name, &value, PROP_TYPE_DRIVE);

    return 0;

}
