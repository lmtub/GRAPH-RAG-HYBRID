static void assign_failed_examine(AssignedDevice *dev)

{

    char name[PATH_MAX], dir[PATH_MAX], driver[PATH_MAX] = {}, *ns;

    uint16_t vendor_id, device_id;

    int r;



    snprintf(dir, sizeof(dir), "/sys/bus/pci/devices/%04x:%02x:%02x.%01x/",

            dev->host.domain, dev->host.bus, dev->host.slot,

            dev->host.function);



    snprintf(name, sizeof(name), "%sdriver", dir);



    r = readlink(name, driver, sizeof(driver));

    if ((r <= 0) || r >= sizeof(driver)) {

        goto fail;

    }




    ns = strrchr(driver, '/');

    if (!ns) {

        goto fail;

    }



    ns++;



    if (get_real_vendor_id(dir, &vendor_id) ||

        get_real_device_id(dir, &device_id)) {

        goto fail;

    }



    error_printf("*** The driver '%s' is occupying your device "

        "%04x:%02x:%02x.%x.\n"

        "***\n"

        "*** You can try the following commands to free it:\n"

        "***\n"

        "*** $ echo \"%04x %04x\" > /sys/bus/pci/drivers/pci-stub/new_id\n"

        "*** $ echo \"%04x:%02x:%02x.%x\" > /sys/bus/pci/drivers/%s/unbind\n"

        "*** $ echo \"%04x:%02x:%02x.%x\" > /sys/bus/pci/drivers/"

        "pci-stub/bind\n"

        "*** $ echo \"%04x %04x\" > /sys/bus/pci/drivers/pci-stub/remove_id\n"

        "***",

        ns, dev->host.domain, dev->host.bus, dev->host.slot,

        dev->host.function, vendor_id, device_id,

        dev->host.domain, dev->host.bus, dev->host.slot, dev->host.function,

        ns, dev->host.domain, dev->host.bus, dev->host.slot,

        dev->host.function, vendor_id, device_id);



    return;



fail:

    error_report("Couldn't find out why.");

}