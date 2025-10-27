void drive_hot_add(Monitor *mon, const QDict *qdict)

{

    int dom, pci_bus;

    unsigned slot;

    int type, bus;

    int success = 0;

    PCIDevice *dev;

    DriveInfo *dinfo;

    const char *pci_addr = qdict_get_str(qdict, "pci_addr");

    const char *opts = qdict_get_str(qdict, "opts");

    BusState *scsibus;



    if (pci_read_devaddr(mon, pci_addr, &dom, &pci_bus, &slot)) {

        return;

    }



    dev = pci_find_device(pci_bus, slot, 0);

    if (!dev) {

        monitor_printf(mon, "no pci device with address %s\n", pci_addr);

        return;

    }



    dinfo = add_init_drive(opts);

    if (!dinfo)

        return;

    if (dinfo->devaddr) {

        monitor_printf(mon, "Parameter addr not supported\n");

        return;

    }

    type = dinfo->type;

    bus = drive_get_max_bus (type);



    switch (type) {

    case IF_SCSI:

        success = 1;

        scsibus = LIST_FIRST(&dev->qdev.child_bus);

        scsi_bus_legacy_add_drive(DO_UPCAST(SCSIBus, qbus, scsibus),

                                  dinfo, dinfo->unit);

        break;

    default:

        monitor_printf(mon, "Can't hot-add drive to type %d\n", type);

    }



    if (success)

        monitor_printf(mon, "OK bus %d, unit %d\n",

                       dinfo->bus,

                       dinfo->unit);

    return;

}
