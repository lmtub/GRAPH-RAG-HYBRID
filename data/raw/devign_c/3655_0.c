static void qbus_list_dev(BusState *bus, char *dest, int len)

{

    DeviceState *dev;

    const char *sep = " ";

    int pos = 0;



    pos += snprintf(dest+pos, len-pos, "devices at \"%s\":",

                    bus->name);

    LIST_FOREACH(dev, &bus->children, sibling) {

        pos += snprintf(dest+pos, len-pos, "%s\"%s\"",

                        sep, dev->info->name);

        if (dev->id)

            pos += snprintf(dest+pos, len-pos, "/\"%s\"", dev->id);

        sep = ", ";

    }

}
