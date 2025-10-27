static int parse_hex32(DeviceState *dev, Property *prop, const char *str)

{

    uint32_t *ptr = qdev_get_prop_ptr(dev, prop);



    if (sscanf(str, "%" PRIx32, ptr) != 1)

        return -EINVAL;

    return 0;

}
