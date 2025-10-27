static int usb_host_scan(void *opaque, USBScanFunc *func)

{

    FILE *f = 0;

    DIR *dir = 0;

    int ret = 0;

    const char *devices = "/devices";

    const char *opened = "husb: opened %s%s\n";

    const char *fs_type[] = {"unknown", "proc", "dev", "sys"};

    char devpath[PATH_MAX];



    /* only check the host once */

    if (!usb_fs_type) {

        f = fopen(USBPROCBUS_PATH "/devices", "r");

        if (f) {

            /* devices found in /proc/bus/usb/ */

            strcpy(devpath, USBPROCBUS_PATH);

            usb_fs_type = USB_FS_PROC;

            fclose(f);

            dprintf(opened, USBPROCBUS_PATH, devices);

        }

        /* try additional methods if an access method hasn't been found yet */

        f = fopen(USBDEVBUS_PATH "/devices", "r");

        if (!usb_fs_type && f) {

            /* devices found in /dev/bus/usb/ */

            strcpy(devpath, USBDEVBUS_PATH);

            usb_fs_type = USB_FS_DEV;

            fclose(f);

            dprintf(opened, USBDEVBUS_PATH, devices);

        }

        dir = opendir(USBSYSBUS_PATH "/devices");

        if (!usb_fs_type && dir) {

            /* devices found in /dev/bus/usb/ (yes - not a mistake!) */

            strcpy(devpath, USBDEVBUS_PATH);

            usb_fs_type = USB_FS_SYS;

            closedir(dir);

            dprintf(opened, USBSYSBUS_PATH, devices);

        }

        if (!usb_fs_type) {

            term_printf("husb: unable to access USB devices\n");

            goto the_end;

        }



        /* the module setting (used later for opening devices) */

        usb_host_device_path = qemu_mallocz(strlen(devpath)+1);

        if (usb_host_device_path) {

            strcpy(usb_host_device_path, devpath);

            term_printf("husb: using %s file-system with %s\n", fs_type[usb_fs_type], usb_host_device_path);

        } else {

            /* out of memory? */

            perror("husb: unable to allocate memory for device path");

            goto the_end;

        }

    }



    switch (usb_fs_type) {

    case USB_FS_PROC:

    case USB_FS_DEV:

        ret = usb_host_scan_dev(opaque, func);

        break;

    case USB_FS_SYS:

        ret = usb_host_scan_sys(opaque, func);

        break;

    }

 the_end:

    return ret;

}
