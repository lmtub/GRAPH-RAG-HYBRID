static PCIDevice *do_pci_register_device(PCIDevice *pci_dev, PCIBus *bus,

                                         const char *name, int devfn,

                                         Error **errp)

{

    PCIDeviceClass *pc = PCI_DEVICE_GET_CLASS(pci_dev);

    PCIConfigReadFunc *config_read = pc->config_read;

    PCIConfigWriteFunc *config_write = pc->config_write;

    Error *local_err = NULL;

    AddressSpace *dma_as;

    DeviceState *dev = DEVICE(pci_dev);



    pci_dev->bus = bus;

    /* Only pci bridges can be attached to extra PCI root buses */

    if (pci_bus_is_root(bus) && bus->parent_dev && !pc->is_bridge) {

        error_setg(errp,

                   "PCI: Only PCI/PCIe bridges can be plugged into %s",

                    bus->parent_dev->name);

        return NULL;

    }



    if (devfn < 0) {

        for(devfn = bus->devfn_min ; devfn < ARRAY_SIZE(bus->devices);

            devfn += PCI_FUNC_MAX) {

            if (!bus->devices[devfn])

                goto found;

        }

        error_setg(errp, "PCI: no slot/function available for %s, all in use",

                   name);

        return NULL;

    found: ;

    } else if (bus->devices[devfn]) {

        error_setg(errp, "PCI: slot %d function %d not available for %s,"

                   " in use by %s",

                   PCI_SLOT(devfn), PCI_FUNC(devfn), name,

                   bus->devices[devfn]->name);

        return NULL;

    } else if (dev->hotplugged &&

               pci_get_function_0(pci_dev)) {

        error_setg(errp, "PCI: slot %d function 0 already ocuppied by %s,"

                   " new func %s cannot be exposed to guest.",

                   PCI_SLOT(devfn),

                   bus->devices[PCI_DEVFN(PCI_SLOT(devfn), 0)]->name,

                   name);



       return NULL;

    }



    pci_dev->devfn = devfn;


    dma_as = pci_device_iommu_address_space(pci_dev);



    memory_region_init_alias(&pci_dev->bus_master_enable_region,

                             OBJECT(pci_dev), "bus master",

                             dma_as->root, 0, memory_region_size(dma_as->root));

    memory_region_set_enabled(&pci_dev->bus_master_enable_region, false);

    address_space_init(&pci_dev->bus_master_as, &pci_dev->bus_master_enable_region,

                       name);



    pstrcpy(pci_dev->name, sizeof(pci_dev->name), name);

    pci_dev->irq_state = 0;

    pci_config_alloc(pci_dev);



    pci_config_set_vendor_id(pci_dev->config, pc->vendor_id);

    pci_config_set_device_id(pci_dev->config, pc->device_id);

    pci_config_set_revision(pci_dev->config, pc->revision);

    pci_config_set_class(pci_dev->config, pc->class_id);



    if (!pc->is_bridge) {

        if (pc->subsystem_vendor_id || pc->subsystem_id) {

            pci_set_word(pci_dev->config + PCI_SUBSYSTEM_VENDOR_ID,

                         pc->subsystem_vendor_id);

            pci_set_word(pci_dev->config + PCI_SUBSYSTEM_ID,

                         pc->subsystem_id);

        } else {

            pci_set_default_subsystem_id(pci_dev);

        }

    } else {

        /* subsystem_vendor_id/subsystem_id are only for header type 0 */

        assert(!pc->subsystem_vendor_id);

        assert(!pc->subsystem_id);

    }

    pci_init_cmask(pci_dev);

    pci_init_wmask(pci_dev);

    pci_init_w1cmask(pci_dev);

    if (pc->is_bridge) {

        pci_init_mask_bridge(pci_dev);

    }

    pci_init_multifunction(bus, pci_dev, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        do_pci_unregister_device(pci_dev);

        return NULL;

    }



    if (!config_read)

        config_read = pci_default_read_config;

    if (!config_write)

        config_write = pci_default_write_config;

    pci_dev->config_read = config_read;

    pci_dev->config_write = config_write;

    bus->devices[devfn] = pci_dev;

    pci_dev->version_id = 2; /* Current pci device vmstate version */

    return pci_dev;

}