static void spapr_vio_busdev_realize(DeviceState *qdev, Error **errp)

{

    sPAPRMachineState *spapr = SPAPR_MACHINE(qdev_get_machine());

    VIOsPAPRDevice *dev = (VIOsPAPRDevice *)qdev;

    VIOsPAPRDeviceClass *pc = VIO_SPAPR_DEVICE_GET_CLASS(dev);

    char *id;



    if (dev->reg != -1) {

        /*

         * Explicitly assigned address, just verify that no-one else

         * is using it.  other mechanism). We have to open code this

         * rather than using spapr_vio_find_by_reg() because sdev

         * itself is already in the list.

         */

        VIOsPAPRDevice *other = reg_conflict(dev);



        if (other) {

            error_setg(errp, "%s and %s devices conflict at address %#x",

                       object_get_typename(OBJECT(qdev)),

                       object_get_typename(OBJECT(&other->qdev)),

                       dev->reg);

            return;

        }

    } else {

        /* Need to assign an address */

        VIOsPAPRBus *bus = SPAPR_VIO_BUS(dev->qdev.parent_bus);



        do {

            dev->reg = bus->next_reg++;

        } while (reg_conflict(dev));

    }



    /* Don't overwrite ids assigned on the command line */

    if (!dev->qdev.id) {

        id = spapr_vio_get_dev_name(DEVICE(dev));

        dev->qdev.id = id;

    }



    dev->irq = xics_alloc(spapr->icp, 0, dev->irq, false);

    if (!dev->irq) {

        error_setg(errp, "can't allocate IRQ");

        return;

    }



    if (pc->rtce_window_size) {

        uint32_t liobn = SPAPR_VIO_LIOBN(dev->reg);



        memory_region_init(&dev->mrroot, OBJECT(dev), "iommu-spapr-root",

                           ram_size);

        memory_region_init_alias(&dev->mrbypass, OBJECT(dev),

                                 "iommu-spapr-bypass", get_system_memory(),

                                 0, ram_size);

        memory_region_add_subregion_overlap(&dev->mrroot, 0, &dev->mrbypass, 1);

        address_space_init(&dev->as, &dev->mrroot, qdev->id);



        dev->tcet = spapr_tce_new_table(qdev, liobn,

                                        0,

                                        SPAPR_TCE_PAGE_SHIFT,

                                        pc->rtce_window_size >>

                                        SPAPR_TCE_PAGE_SHIFT, false);

        dev->tcet->vdev = dev;

        memory_region_add_subregion_overlap(&dev->mrroot, 0,

                                            spapr_tce_get_iommu(dev->tcet), 2);

    }



    pc->realize(dev, errp);

}
