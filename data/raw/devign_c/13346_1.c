static int parse_chr(DeviceState *dev, Property *prop, const char *str)

{

    CharDriverState **ptr = qdev_get_prop_ptr(dev, prop);



    *ptr = qemu_chr_find(str);

    if (*ptr == NULL) {

        return -ENOENT;

    }

    if ((*ptr)->assigned) {

        return -EEXIST;

    }

    (*ptr)->assigned = 1;

    return 0;

}
