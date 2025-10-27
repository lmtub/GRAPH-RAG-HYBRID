static PCIDevice *qemu_pci_hot_add_storage(Monitor *mon,

                                           const char *devaddr,

                                           const char *opts)

{

    PCIDevice *dev;

    DriveInfo *dinfo = NULL;

    int type = -1;

    char buf[128];



    if (get_param_value(buf, sizeof(buf), "if", opts)) {

        if (!strcmp(buf, "scsi"))

            type = IF_SCSI;

        else if (!strcmp(buf, "virtio")) {

            type = IF_VIRTIO;

        } else {

            monitor_printf(mon, "type %s not a hotpluggable PCI device.\n", buf);

            return NULL;

        }

    } else {

        monitor_printf(mon, "no if= specified\n");

        return NULL;

    }



    if (get_param_value(buf, sizeof(buf), "file", opts)) {

        dinfo = add_init_drive(opts);

        if (!dinfo)

            return NULL;

        if (dinfo->devaddr) {

            monitor_printf(mon, "Parameter addr not supported\n");

            return NULL;

        }

    } else {

        dinfo = NULL;

    }



    switch (type) {

    case IF_SCSI:

        dev = pci_create("lsi53c895a", devaddr);

        break;

    case IF_VIRTIO:

        if (!dinfo) {

            monitor_printf(mon, "virtio requires a backing file/device.\n");

            return NULL;

        }

        dev = pci_create("virtio-blk-pci", devaddr);

        qdev_prop_set_drive(&dev->qdev, "drive", dinfo);

        break;

    default:

        dev = NULL;

    }

    if (dev)

        qdev_init(&dev->qdev);

    return dev;

}
