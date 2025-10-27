static void spapr_pci_unplug_request(HotplugHandler *plug_handler,

                                     DeviceState *plugged_dev, Error **errp)

{

    sPAPRPHBState *phb = SPAPR_PCI_HOST_BRIDGE(DEVICE(plug_handler));

    PCIDevice *pdev = PCI_DEVICE(plugged_dev);

    sPAPRDRConnectorClass *drck;

    sPAPRDRConnector *drc = spapr_phb_get_pci_drc(phb, pdev);



    if (!phb->dr_enabled) {

        error_setg(errp, QERR_BUS_NO_HOTPLUG,

                   object_get_typename(OBJECT(phb)));

        return;

    }



    g_assert(drc);

    g_assert(drc->dev == plugged_dev);



    drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);

    if (!drck->release_pending(drc)) {

        PCIBus *bus = PCI_BUS(qdev_get_parent_bus(DEVICE(pdev)));

        uint32_t slotnr = PCI_SLOT(pdev->devfn);

        sPAPRDRConnector *func_drc;

        sPAPRDRConnectorClass *func_drck;

        sPAPRDREntitySense state;

        int i;



        /* ensure any other present functions are pending unplug */

        if (PCI_FUNC(pdev->devfn) == 0) {

            for (i = 1; i < 8; i++) {

                func_drc = spapr_phb_get_pci_func_drc(phb, pci_bus_num(bus),

                                                      PCI_DEVFN(slotnr, i));

                func_drck = SPAPR_DR_CONNECTOR_GET_CLASS(func_drc);

                state = func_drck->dr_entity_sense(func_drc);

                if (state == SPAPR_DR_ENTITY_SENSE_PRESENT

                    && !func_drck->release_pending(func_drc)) {

                    error_setg(errp,

                               "PCI: slot %d, function %d still present. "

                               "Must unplug all non-0 functions first.",

                               slotnr, i);

                    return;

                }

            }

        }



        spapr_drc_detach(drc);



        /* if this isn't func 0, defer unplug event. otherwise signal removal

         * for all present functions

         */

        if (PCI_FUNC(pdev->devfn) == 0) {

            for (i = 7; i >= 0; i--) {

                func_drc = spapr_phb_get_pci_func_drc(phb, pci_bus_num(bus),

                                                      PCI_DEVFN(slotnr, i));

                func_drck = SPAPR_DR_CONNECTOR_GET_CLASS(func_drc);

                state = func_drck->dr_entity_sense(func_drc);

                if (state == SPAPR_DR_ENTITY_SENSE_PRESENT) {

                    spapr_hotplug_req_remove_by_index(func_drc);

                }

            }

        }

    }

}
