static void qbus_list_bus(DeviceState *dev, char *dest, int len)

{

    BusState *child;

    const char *sep = " ";

    int pos = 0;



    pos += snprintf(dest+pos, len-pos,"child busses at \"%s\":",

                    dev->id ? dev->id : dev->info->name);

    LIST_FOREACH(child, &dev->child_bus, sibling) {

        pos += snprintf(dest+pos, len-pos, "%s\"%s\"", sep, child->name);

        sep = ", ";

    }

}
