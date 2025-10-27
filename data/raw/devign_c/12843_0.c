static void spapr_phb_realize(DeviceState *dev, Error **errp)

{

    sPAPRMachineState *spapr = SPAPR_MACHINE(qdev_get_machine());

    SysBusDevice *s = SYS_BUS_DEVICE(dev);

    sPAPRPHBState *sphb = SPAPR_PCI_HOST_BRIDGE(s);

    PCIHostState *phb = PCI_HOST_BRIDGE(s);

    char *namebuf;

    int i;

    PCIBus *bus;

    uint64_t msi_window_size = 4096;

    sPAPRTCETable *tcet;

    const unsigned windows_supported =

        sphb->ddw_enabled ? SPAPR_PCI_DMA_MAX_WINDOWS : 1;



    if (sphb->index != (uint32_t)-1) {

        sPAPRMachineClass *smc = SPAPR_MACHINE_GET_CLASS(spapr);

        Error *local_err = NULL;



        if ((sphb->buid != (uint64_t)-1) || (sphb->dma_liobn[0] != (uint32_t)-1)

            || (sphb->dma_liobn[1] != (uint32_t)-1 && windows_supported == 2)

            || (sphb->mem_win_addr != (hwaddr)-1)

            || (sphb->mem64_win_addr != (hwaddr)-1)

            || (sphb->io_win_addr != (hwaddr)-1)) {

            error_setg(errp, "Either \"index\" or other parameters must"

                       " be specified for PAPR PHB, not both");

            return;

        }



        smc->phb_placement(spapr, sphb->index,

                           &sphb->buid, &sphb->io_win_addr,

                           &sphb->mem_win_addr, &sphb->mem64_win_addr,

                           windows_supported, sphb->dma_liobn, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            return;

        }

    }



    if (sphb->buid == (uint64_t)-1) {

        error_setg(errp, "BUID not specified for PHB");

        return;

    }



    if ((sphb->dma_liobn[0] == (uint32_t)-1) ||

        ((sphb->dma_liobn[1] == (uint32_t)-1) && (windows_supported > 1))) {

        error_setg(errp, "LIOBN(s) not specified for PHB");

        return;

    }



    if (sphb->mem_win_addr == (hwaddr)-1) {

        error_setg(errp, "Memory window address not specified for PHB");

        return;

    }



    if (sphb->io_win_addr == (hwaddr)-1) {

        error_setg(errp, "IO window address not specified for PHB");

        return;

    }



    if (sphb->mem64_win_size != 0) {

        if (sphb->mem64_win_addr == (hwaddr)-1) {

            error_setg(errp,

                       "64-bit memory window address not specified for PHB");

            return;

        }



        if (sphb->mem_win_size > SPAPR_PCI_MEM32_WIN_SIZE) {

            error_setg(errp, "32-bit memory window of size 0x%"HWADDR_PRIx

                       " (max 2 GiB)", sphb->mem_win_size);

            return;

        }



        if (sphb->mem64_win_pciaddr == (hwaddr)-1) {

            /* 64-bit window defaults to identity mapping */

            sphb->mem64_win_pciaddr = sphb->mem64_win_addr;

        }

    } else if (sphb->mem_win_size > SPAPR_PCI_MEM32_WIN_SIZE) {

        /*

         * For compatibility with old configuration, if no 64-bit MMIO

         * window is specified, but the ordinary (32-bit) memory

         * window is specified as > 2GiB, we treat it as a 2GiB 32-bit

         * window, with a 64-bit MMIO window following on immediately

         * afterwards

         */

        sphb->mem64_win_size = sphb->mem_win_size - SPAPR_PCI_MEM32_WIN_SIZE;

        sphb->mem64_win_addr = sphb->mem_win_addr + SPAPR_PCI_MEM32_WIN_SIZE;

        sphb->mem64_win_pciaddr =

            SPAPR_PCI_MEM_WIN_BUS_OFFSET + SPAPR_PCI_MEM32_WIN_SIZE;

        sphb->mem_win_size = SPAPR_PCI_MEM32_WIN_SIZE;

    }



    if (spapr_pci_find_phb(spapr, sphb->buid)) {

        error_setg(errp, "PCI host bridges must have unique BUIDs");

        return;

    }



    if (sphb->numa_node != -1 &&

        (sphb->numa_node >= MAX_NODES || !numa_info[sphb->numa_node].present)) {

        error_setg(errp, "Invalid NUMA node ID for PCI host bridge");

        return;

    }



    sphb->dtbusname = g_strdup_printf("pci@%" PRIx64, sphb->buid);



    /* Initialize memory regions */

    namebuf = g_strdup_printf("%s.mmio", sphb->dtbusname);

    memory_region_init(&sphb->memspace, OBJECT(sphb), namebuf, UINT64_MAX);

    g_free(namebuf);



    namebuf = g_strdup_printf("%s.mmio32-alias", sphb->dtbusname);

    memory_region_init_alias(&sphb->mem32window, OBJECT(sphb),

                             namebuf, &sphb->memspace,

                             SPAPR_PCI_MEM_WIN_BUS_OFFSET, sphb->mem_win_size);

    g_free(namebuf);

    memory_region_add_subregion(get_system_memory(), sphb->mem_win_addr,

                                &sphb->mem32window);



    if (sphb->mem64_win_pciaddr != (hwaddr)-1) {

        namebuf = g_strdup_printf("%s.mmio64-alias", sphb->dtbusname);

        memory_region_init_alias(&sphb->mem64window, OBJECT(sphb),

                                 namebuf, &sphb->memspace,

                                 sphb->mem64_win_pciaddr, sphb->mem64_win_size);

        g_free(namebuf);



        if (sphb->mem64_win_addr != (hwaddr)-1) {

            memory_region_add_subregion(get_system_memory(),

                                        sphb->mem64_win_addr,

                                        &sphb->mem64window);

        }

    }



    /* Initialize IO regions */

    namebuf = g_strdup_printf("%s.io", sphb->dtbusname);

    memory_region_init(&sphb->iospace, OBJECT(sphb),

                       namebuf, SPAPR_PCI_IO_WIN_SIZE);

    g_free(namebuf);



    namebuf = g_strdup_printf("%s.io-alias", sphb->dtbusname);

    memory_region_init_alias(&sphb->iowindow, OBJECT(sphb), namebuf,

                             &sphb->iospace, 0, SPAPR_PCI_IO_WIN_SIZE);

    g_free(namebuf);

    memory_region_add_subregion(get_system_memory(), sphb->io_win_addr,

                                &sphb->iowindow);



    bus = pci_register_bus(dev, NULL,

                           pci_spapr_set_irq, pci_spapr_map_irq, sphb,

                           &sphb->memspace, &sphb->iospace,

                           PCI_DEVFN(0, 0), PCI_NUM_PINS, TYPE_PCI_BUS);

    phb->bus = bus;

    qbus_set_hotplug_handler(BUS(phb->bus), DEVICE(sphb), NULL);



    /*

     * Initialize PHB address space.

     * By default there will be at least one subregion for default

     * 32bit DMA window.

     * Later the guest might want to create another DMA window

     * which will become another memory subregion.

     */

    namebuf = g_strdup_printf("%s.iommu-root", sphb->dtbusname);

    memory_region_init(&sphb->iommu_root, OBJECT(sphb),

                       namebuf, UINT64_MAX);

    g_free(namebuf);

    address_space_init(&sphb->iommu_as, &sphb->iommu_root,

                       sphb->dtbusname);



    /*

     * As MSI/MSIX interrupts trigger by writing at MSI/MSIX vectors,

     * we need to allocate some memory to catch those writes coming

     * from msi_notify()/msix_notify().

     * As MSIMessage:addr is going to be the same and MSIMessage:data

     * is going to be a VIRQ number, 4 bytes of the MSI MR will only

     * be used.

     *

     * For KVM we want to ensure that this memory is a full page so that

     * our memory slot is of page size granularity.

     */

#ifdef CONFIG_KVM

    if (kvm_enabled()) {

        msi_window_size = getpagesize();

    }

#endif



    memory_region_init_io(&sphb->msiwindow, OBJECT(sphb), &spapr_msi_ops, spapr,

                          "msi", msi_window_size);

    memory_region_add_subregion(&sphb->iommu_root, SPAPR_PCI_MSI_WINDOW,

                                &sphb->msiwindow);



    pci_setup_iommu(bus, spapr_pci_dma_iommu, sphb);



    pci_bus_set_route_irq_fn(bus, spapr_route_intx_pin_to_irq);



    QLIST_INSERT_HEAD(&spapr->phbs, sphb, list);



    /* Initialize the LSI table */

    for (i = 0; i < PCI_NUM_PINS; i++) {

        uint32_t irq;

        Error *local_err = NULL;



        irq = spapr_ics_alloc_block(spapr->ics, 1, true, false, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            error_prepend(errp, "can't allocate LSIs: ");

            return;

        }



        sphb->lsi_table[i].irq = irq;

    }



    /* allocate connectors for child PCI devices */

    if (sphb->dr_enabled) {

        for (i = 0; i < PCI_SLOT_MAX * 8; i++) {

            spapr_dr_connector_new(OBJECT(phb), TYPE_SPAPR_DRC_PCI,

                                   (sphb->index << 16) | i);

        }

    }



    /* DMA setup */

    if (((sphb->page_size_mask & qemu_getrampagesize()) == 0)

        && kvm_enabled()) {

        error_report("System page size 0x%lx is not enabled in page_size_mask "

                     "(0x%"PRIx64"). Performance may be slow",

                     qemu_getrampagesize(), sphb->page_size_mask);

    }



    for (i = 0; i < windows_supported; ++i) {

        tcet = spapr_tce_new_table(DEVICE(sphb), sphb->dma_liobn[i]);

        if (!tcet) {

            error_setg(errp, "Creating window#%d failed for %s",

                       i, sphb->dtbusname);

            return;

        }

        memory_region_add_subregion(&sphb->iommu_root, 0,

                                    spapr_tce_get_iommu(tcet));

    }



    sphb->msi = g_hash_table_new_full(g_int_hash, g_int_equal, g_free, g_free);

}
