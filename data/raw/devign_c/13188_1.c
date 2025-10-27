static void isabus_fdc_realize(DeviceState *dev, Error **errp)

{

    ISADevice *isadev = ISA_DEVICE(dev);

    FDCtrlISABus *isa = ISA_FDC(dev);

    FDCtrl *fdctrl = &isa->state;

    Error *err = NULL;



    isa_register_portio_list(isadev, isa->iobase, fdc_portio_list, fdctrl,

                             "fdc");



    isa_init_irq(isadev, &fdctrl->irq, isa->irq);

    fdctrl->dma_chann = isa->dma;

    if (fdctrl->dma_chann != -1) {

        fdctrl->dma = isa_get_dma(isa_bus_from_device(isadev), isa->dma);

        assert(fdctrl->dma);

    }



    qdev_set_legacy_instance_id(dev, isa->iobase, 2);

    fdctrl_realize_common(fdctrl, &err);

    if (err != NULL) {

        error_propagate(errp, err);

        return;

    }

}
