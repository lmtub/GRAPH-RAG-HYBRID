int pci_piix3_xen_ide_unplug(DeviceState *dev)

{

    PCIIDEState *pci_ide;

    DriveInfo *di;

    int i;

    IDEDevice *idedev;



    pci_ide = PCI_IDE(dev);



    for (i = 0; i < 4; i++) {

        di = drive_get_by_index(IF_IDE, i);

        if (di != NULL && !di->media_cd) {

            BlockBackend *blk = blk_by_legacy_dinfo(di);

            DeviceState *ds = blk_get_attached_dev(blk);



            blk_drain(blk);

            blk_flush(blk);



            if (ds) {

                blk_detach_dev(blk, ds);

            }

            pci_ide->bus[di->bus].ifs[di->unit].blk = NULL;

            if (!(i % 2)) {

                idedev = pci_ide->bus[di->bus].master;

            } else {

                idedev = pci_ide->bus[di->bus].slave;

            }

            idedev->conf.blk = NULL;

            monitor_remove_blk(blk);

            blk_unref(blk);

        }

    }

    qdev_reset_all(DEVICE(dev));

    return 0;

}
