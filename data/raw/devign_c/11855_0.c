static int ide_dev_initfn(IDEDevice *dev, IDEDriveKind kind)

{

    IDEBus *bus = DO_UPCAST(IDEBus, qbus, dev->qdev.parent_bus);

    IDEState *s = bus->ifs + dev->unit;

    const char *serial;

    DriveInfo *dinfo;



    if (dev->conf.discard_granularity && dev->conf.discard_granularity != 512) {

        error_report("discard_granularity must be 512 for ide");

        return -1;

    }



    serial = dev->serial;

    if (!serial) {

        /* try to fall back to value set with legacy -drive serial=... */

        dinfo = drive_get_by_blockdev(dev->conf.bs);

        if (*dinfo->serial) {

            serial = dinfo->serial;

        }

    }



    if (ide_init_drive(s, dev->conf.bs, kind, dev->version, serial) < 0) {

        return -1;

    }



    if (!dev->version) {

        dev->version = g_strdup(s->version);

    }

    if (!dev->serial) {

        dev->serial = g_strdup(s->drive_serial_str);

    }



    add_boot_device_path(dev->conf.bootindex, &dev->qdev,

                         dev->unit ? "/disk@1" : "/disk@0");



    return 0;

}
