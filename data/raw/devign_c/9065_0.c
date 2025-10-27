static int usb_host_scan(void *opaque, USBScanFunc *func)

{

    Monitor *mon = cur_mon;

    FILE *f = NULL;

    DIR *dir = NULL;

    int ret = 0;

    const char *fs_type[] = {"unknown", "proc", "dev", "sys"};

    char devpath[PATH_MAX];



    /* only check the host once */

    if (!usb_fs_type) {

        dir = opendir(USBSYSBUS_PATH "/devices");

        if (dir) {

            /* devices found in /dev/bus/usb/ (yes - not a mistake!) */

            strcpy(devpath, USBDEVBUS_PATH);

            usb_fs_type = USB_FS_SYS;

            closedir(dir);

            dprintf(USBDBG_DEVOPENED, USBSYSBUS_PATH);

            goto found_devices;

        }

        f = fopen(USBPROCBUS_PATH "/devices", "r");

        if (f) {

            /* devices found in /proc/bus/usb/ */

            strcpy(devpath, USBPROCBUS_PATH);

            usb_fs_type = USB_FS_PROC;

            fclose(f);

            dprintf(USBDBG_DEVOPENED, USBPROCBUS_PATH);

            goto found_devices;

        }

        /* try additional methods if an access method hasn't been found yet */

        f = fopen(USBDEVBUS_PATH "/devices", "r");

        if (f) {

            /* devices found in /dev/bus/usb/ */

            strcpy(devpath, USBDEVBUS_PATH);

            usb_fs_type = USB_FS_DEV;

            fclose(f);

            dprintf(USBDBG_DEVOPENED, USBDEVBUS_PATH);

            goto found_devices;

        }

    found_devices:

        if (!usb_fs_type) {

            monitor_printf(mon, "husb: unable to access USB devices\n");

            return -ENOENT;

        }



        /* the module setting (used later for opening devices) */

        usb_host_device_path = qemu_mallocz(strlen(devpath)+1);

        strcpy(usb_host_device_path, devpath);

        monitor_printf(mon, "husb: using %s file-system with %s\n",

                       fs_type[usb_fs_type], usb_host_device_path);

    }



    switch (usb_fs_type) {

    case USB_FS_PROC:

    case USB_FS_DEV:

        ret = usb_host_scan_dev(opaque, func);

        break;

    case USB_FS_SYS:

        ret = usb_host_scan_sys(opaque, func);

        break;

    default:

        ret = -EINVAL;

        break;

    }

    return ret;

}
