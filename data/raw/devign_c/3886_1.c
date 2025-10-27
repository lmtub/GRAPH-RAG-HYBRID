static int spapr_vio_busdev_init(DeviceState *qdev)

{

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

            fprintf(stderr, "vio: %s and %s devices conflict at address %#x\n",

                    object_get_typename(OBJECT(qdev)),

                    object_get_typename(OBJECT(&other->qdev)),

                    dev->reg);

            return -1;

        }

    } else {

        /* Need to assign an address */

        VIOsPAPRBus *bus = DO_UPCAST(VIOsPAPRBus, bus, dev->qdev.parent_bus);



        do {

            dev->reg = bus->next_reg++;

        } while (reg_conflict(dev));

    }



    /* Don't overwrite ids assigned on the command line */

    if (!dev->qdev.id) {

        id = vio_format_dev_name(dev);

        if (!id) {

            return -1;

        }

        dev->qdev.id = id;

    }



    dev->qirq = spapr_allocate_msi(dev->vio_irq_num, &dev->vio_irq_num);

    if (!dev->qirq) {

        return -1;

    }



    rtce_init(dev);



    return pc->init(dev);

}
