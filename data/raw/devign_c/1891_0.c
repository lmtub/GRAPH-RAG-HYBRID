static int print_ptr(DeviceState *dev, Property *prop, char *dest, size_t len)

{

    void **ptr = qdev_get_prop_ptr(dev, prop);

    return snprintf(dest, len, "<%p>", *ptr);

}
