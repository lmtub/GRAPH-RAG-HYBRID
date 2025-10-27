static void spapr_phb_hot_plug_child(HotplugHandler *plug_handler,

                                     DeviceState *plugged_dev, Error **errp)

{

    sPAPRPHBState *phb = SPAPR_PCI_HOST_BRIDGE(DEVICE(plug_handler));

    PCIDevice *pdev = PCI_DEVICE(plugged_dev);

    sPAPRDRConnector *drc = spapr_phb_get_pci_drc(phb, pdev);

    Error *local_err = NULL;



    /* if DR is disabled we don't need to do anything in the case of

     * hotplug or coldplug callbacks

     */

    if (!phb->dr_enabled) {

        /* if this is a hotplug operation initiated by the user

         * we need to let them know it's not enabled

         */

        if (plugged_dev->hotplugged) {

            error_setg(errp, QERR_BUS_NO_HOTPLUG,

                       object_get_typename(OBJECT(phb)));

        }

        return;

    }



    g_assert(drc);



    spapr_phb_add_pci_device(drc, phb, pdev, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }

    if (plugged_dev->hotplugged) {

        spapr_hotplug_req_add_by_index(drc);

    }

}
