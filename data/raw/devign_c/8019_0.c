static int print_bit(DeviceState *dev, Property *prop, char *dest, size_t len)

{

    uint32_t *p = qdev_get_prop_ptr(dev, prop);

    return snprintf(dest, len, (*p & qdev_get_prop_mask(prop)) ? "on" : "off");

}
