static int print_size(DeviceState *dev, Property *prop, char *dest, size_t len)

{

    uint64_t *ptr = qdev_get_prop_ptr(dev, prop);

    char suffixes[] = {'T', 'G', 'M', 'K', 'B'};

    int i = 0;

    uint64_t div;



    for (div = 1ULL << 40; !(*ptr / div) ; div >>= 10) {

        i++;

    }

    return snprintf(dest, len, "%0.03f%c", (double)*ptr/div, suffixes[i]);

}
