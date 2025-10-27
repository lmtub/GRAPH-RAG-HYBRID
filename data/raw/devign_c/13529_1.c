QemuConsole *qemu_console_lookup_by_device(DeviceState *dev, uint32_t head)

{

    Error *local_err = NULL;

    Object *obj;

    uint32_t h;

    int i;



    for (i = 0; i < nb_consoles; i++) {

        if (!consoles[i]) {

            continue;

        }

        obj = object_property_get_link(OBJECT(consoles[i]),

                                       "device", &local_err);

        if (DEVICE(obj) != dev) {

            continue;

        }

        h = object_property_get_int(OBJECT(consoles[i]),

                                    "head", &local_err);

        if (h != head) {

            continue;

        }

        return consoles[i];

    }

    return NULL;

}
