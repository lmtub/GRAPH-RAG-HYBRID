static int scsi_qdev_init(DeviceState *qdev)

{

    SCSIDevice *dev = SCSI_DEVICE(qdev);

    SCSIBus *bus = DO_UPCAST(SCSIBus, qbus, dev->qdev.parent_bus);

    SCSIDevice *d;

    int rc = -1;



    if (dev->channel > bus->info->max_channel) {

        error_report("bad scsi channel id: %d", dev->channel);

        goto err;

    }

    if (dev->id != -1 && dev->id > bus->info->max_target) {

        error_report("bad scsi device id: %d", dev->id);

        goto err;

    }



    if (dev->id == -1) {

        int id = -1;

        if (dev->lun == -1) {

            dev->lun = 0;

        }

        do {

            d = scsi_device_find(bus, dev->channel, ++id, dev->lun);

        } while (d && d->lun == dev->lun && id <= bus->info->max_target);

        if (id > bus->info->max_target) {

            error_report("no free target");

            goto err;

        }

        dev->id = id;

    } else if (dev->lun == -1) {

        int lun = -1;

        do {

            d = scsi_device_find(bus, dev->channel, dev->id, ++lun);

        } while (d && d->lun == lun && lun < bus->info->max_lun);

        if (lun > bus->info->max_lun) {

            error_report("no free lun");

            goto err;

        }

        dev->lun = lun;

    } else {

        d = scsi_device_find(bus, dev->channel, dev->id, dev->lun);

        if (dev->lun == d->lun && dev != d) {

            qdev_free(&d->qdev);

        }

    }



    QTAILQ_INIT(&dev->requests);

    rc = scsi_device_init(dev);

    if (rc == 0) {

        dev->vmsentry = qemu_add_vm_change_state_handler(scsi_dma_restart_cb,

                                                         dev);

    }



err:

    return rc;

}
