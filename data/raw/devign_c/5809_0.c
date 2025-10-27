static IOMMUTLBEntry s390_translate_iommu(MemoryRegion *iommu, hwaddr addr,

                                          bool is_write)

{

    uint64_t pte;

    uint32_t flags;

    S390PCIBusDevice *pbdev = container_of(iommu, S390PCIBusDevice, mr);

    S390pciState *s;

    IOMMUTLBEntry ret = {

        .target_as = &address_space_memory,

        .iova = 0,

        .translated_addr = 0,

        .addr_mask = ~(hwaddr)0,

        .perm = IOMMU_NONE,

    };



    if (!pbdev->configured || !pbdev->pdev) {

        return ret;

    }



    DPRINTF("iommu trans addr 0x%" PRIx64 "\n", addr);



    s = S390_PCI_HOST_BRIDGE(pci_device_root_bus(pbdev->pdev)->qbus.parent);

    /* s390 does not have an APIC mapped to main storage so we use

     * a separate AddressSpace only for msix notifications

     */

    if (addr == ZPCI_MSI_ADDR) {

        ret.target_as = &s->msix_notify_as;

        ret.iova = addr;

        ret.translated_addr = addr;

        ret.addr_mask = 0xfff;

        ret.perm = IOMMU_RW;

        return ret;

    }



    if (!pbdev->g_iota) {

        pbdev->error_state = true;

        pbdev->lgstg_blocked = true;

        s390_pci_generate_error_event(ERR_EVENT_INVALAS, pbdev->fh, pbdev->fid,

                                      addr, 0);

        return ret;

    }



    if (addr < pbdev->pba || addr > pbdev->pal) {

        pbdev->error_state = true;

        pbdev->lgstg_blocked = true;

        s390_pci_generate_error_event(ERR_EVENT_OORANGE, pbdev->fh, pbdev->fid,

                                      addr, 0);

        return ret;

    }



    pte = s390_guest_io_table_walk(s390_pci_get_table_origin(pbdev->g_iota),

                                   addr);



    if (!pte) {

        pbdev->error_state = true;

        pbdev->lgstg_blocked = true;

        s390_pci_generate_error_event(ERR_EVENT_SERR, pbdev->fh, pbdev->fid,

                                      addr, ERR_EVENT_Q_BIT);

        return ret;

    }



    flags = pte & ZPCI_PTE_FLAG_MASK;

    ret.iova = addr;

    ret.translated_addr = pte & ZPCI_PTE_ADDR_MASK;

    ret.addr_mask = 0xfff;



    if (flags & ZPCI_PTE_INVALID) {

        ret.perm = IOMMU_NONE;

    } else {

        ret.perm = IOMMU_RW;

    }



    return ret;

}
