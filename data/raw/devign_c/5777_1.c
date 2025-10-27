static int print_drive(DeviceState *dev, Property *prop, char *dest, size_t len)

{

    DriveInfo **ptr = qdev_get_prop_ptr(dev, prop);

    return snprintf(dest, len, "%s", (*ptr)->id);

}
