static BusState *qbus_find(const char *path)

{

    DeviceState *dev;

    BusState *bus;

    char elem[128];

    int pos, len;



    /* find start element */

    if (path[0] == '/') {

        bus = sysbus_get_default();

        pos = 0;

    } else {

        if (sscanf(path, "%127[^/]%n", elem, &len) != 1) {

            assert(!path[0]);

            elem[0] = len = 0;

        }

        bus = qbus_find_recursive(sysbus_get_default(), elem, NULL);

        if (!bus) {

            qerror_report(QERR_BUS_NOT_FOUND, elem);

            return NULL;

        } else if (qbus_is_full(bus)) {

            qerror_report(ERROR_CLASS_GENERIC_ERROR, "Bus '%s' is full",

                          elem);

            return NULL;

        }

        pos = len;

    }



    for (;;) {

        assert(path[pos] == '/' || !path[pos]);

        while (path[pos] == '/') {

            pos++;

        }

        if (path[pos] == '\0') {

            return bus;

        }



        /* find device */

        if (sscanf(path+pos, "%127[^/]%n", elem, &len) != 1) {

            g_assert_not_reached();

            elem[0] = len = 0;

        }

        pos += len;

        dev = qbus_find_dev(bus, elem);

        if (!dev) {

            qerror_report(QERR_DEVICE_NOT_FOUND, elem);

            if (!monitor_cur_is_qmp()) {

                qbus_list_dev(bus);

            }

            return NULL;

        }



        assert(path[pos] == '/' || !path[pos]);

        while (path[pos] == '/') {

            pos++;

        }

        if (path[pos] == '\0') {

            /* last specified element is a device.  If it has exactly

             * one child bus accept it nevertheless */

            switch (dev->num_child_bus) {

            case 0:

                qerror_report(ERROR_CLASS_GENERIC_ERROR,

                              "Device '%s' has no child bus", elem);

                return NULL;

            case 1:

                return QLIST_FIRST(&dev->child_bus);

            default:

                qerror_report(ERROR_CLASS_GENERIC_ERROR,

                              "Device '%s' has multiple child busses", elem);

                if (!monitor_cur_is_qmp()) {

                    qbus_list_bus(dev);

                }

                return NULL;

            }

        }



        /* find bus */

        if (sscanf(path+pos, "%127[^/]%n", elem, &len) != 1) {

            g_assert_not_reached();

            elem[0] = len = 0;

        }

        pos += len;

        bus = qbus_find_bus(dev, elem);

        if (!bus) {

            qerror_report(QERR_BUS_NOT_FOUND, elem);

            if (!monitor_cur_is_qmp()) {

                qbus_list_bus(dev);

            }

            return NULL;

        }

    }

}
