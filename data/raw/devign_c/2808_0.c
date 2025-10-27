static int usb_host_scan_sys(void *opaque, USBScanFunc *func)

{

    DIR *dir = 0;

    char line[1024];

    int bus_num, addr, speed, class_id, product_id, vendor_id;

    int ret = 0;

    char product_name[512];

    struct dirent *de;



    dir = opendir(USBSYSBUS_PATH "/devices");

    if (!dir) {

        perror("husb: cannot open devices directory");

        goto the_end;

    }



    while ((de = readdir(dir))) {

        if (de->d_name[0] != '.' && !strchr(de->d_name, ':')) {

            char *tmpstr = de->d_name;

            if (!strncmp(de->d_name, "usb", 3))

                tmpstr += 3;

            bus_num = atoi(tmpstr);



            if (!usb_host_read_file(line, sizeof(line), USBSYSBUS_PATH "/devices/%s/devnum", de->d_name))

                goto the_end;

            if (sscanf(line, "%d", &addr) != 1)

                goto the_end;



            if (!usb_host_read_file(line, sizeof(line), USBSYSBUS_PATH "/devices/%s/bDeviceClass", de->d_name))

                goto the_end;

            if (sscanf(line, "%x", &class_id) != 1)

                goto the_end;



            if (!usb_host_read_file(line, sizeof(line), USBSYSBUS_PATH "/devices/%s/idVendor", de->d_name))

                goto the_end;

            if (sscanf(line, "%x", &vendor_id) != 1)

                goto the_end;



            if (!usb_host_read_file(line, sizeof(line), USBSYSBUS_PATH "/devices/%s/idProduct", de->d_name))

                goto the_end;

            if (sscanf(line, "%x", &product_id) != 1)

                goto the_end;



            if (!usb_host_read_file(line, sizeof(line), USBSYSBUS_PATH "/devices/%s/product", de->d_name)) {

                *product_name = 0;

            } else {

                if (strlen(line) > 0)

                    line[strlen(line) - 1] = '\0';

                pstrcpy(product_name, sizeof(product_name), line);

            }



            if (!usb_host_read_file(line, sizeof(line), USBSYSBUS_PATH "/devices/%s/speed", de->d_name))

                goto the_end;

            if (!strcmp(line, "480\n"))

                speed = USB_SPEED_HIGH;

            else if (!strcmp(line, "1.5\n"))

                speed = USB_SPEED_LOW;

            else

                speed = USB_SPEED_FULL;



            ret = func(opaque, bus_num, addr, class_id, vendor_id,

                       product_id, product_name, speed);

            if (ret)

                goto the_end;

        }

    }

 the_end:

    if (dir)

        closedir(dir);

    return ret;

}
