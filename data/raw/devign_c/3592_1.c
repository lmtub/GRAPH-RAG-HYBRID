void usb_host_info(Monitor *mon, const QDict *qdict)

{

    libusb_device **devs;

    struct libusb_device_descriptor ddesc;

    char port[16];

    int i, n;



    if (usb_host_init() != 0) {

        return;

    }



    n = libusb_get_device_list(ctx, &devs);

    for (i = 0; i < n; i++) {

        if (libusb_get_device_descriptor(devs[i], &ddesc) != 0) {

            continue;

        }

        if (ddesc.bDeviceClass == LIBUSB_CLASS_HUB) {

            continue;

        }

        usb_host_get_port(devs[i], port, sizeof(port));

        monitor_printf(mon, "  Bus %d, Addr %d, Port %s, Speed %s Mb/s\n",

                       libusb_get_bus_number(devs[i]),

                       libusb_get_device_address(devs[i]),

                       port,

                       speed_name[libusb_get_device_speed(devs[i])]);

        monitor_printf(mon, "    Class %02x:", ddesc.bDeviceClass);

        monitor_printf(mon, " USB device %04x:%04x",

                       ddesc.idVendor, ddesc.idProduct);

        if (ddesc.iProduct) {

            libusb_device_handle *handle;

            if (libusb_open(devs[i], &handle) == 0) {

                unsigned char name[64] = "";

                libusb_get_string_descriptor_ascii(handle,

                                                   ddesc.iProduct,

                                                   name, sizeof(name));

                libusb_close(handle);

                monitor_printf(mon, ", %s", name);

            }

        }

        monitor_printf(mon, "\n");

    }

    libusb_free_device_list(devs, 1);

}
