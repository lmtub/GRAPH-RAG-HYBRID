void qdev_prop_set_drive(DeviceState *dev, const char *name, DriveInfo *value)

{

    qdev_prop_set(dev, name, &value, PROP_TYPE_DRIVE);

}
