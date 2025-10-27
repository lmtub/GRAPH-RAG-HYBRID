static int mch_init(PCIDevice *d)

{

    int i;

    hwaddr pci_hole64_size;

    MCHPCIState *mch = MCH_PCI_DEVICE(d);



    /* Leave enough space for the biggest MCFG BAR */

    /* TODO: this matches current bios behaviour, but

     * it's not a power of two, which means an MTRR

     * can't cover it exactly.

     */

    mch->guest_info->pci_info.w32.begin = MCH_HOST_BRIDGE_PCIEXBAR_DEFAULT +

        MCH_HOST_BRIDGE_PCIEXBAR_MAX;



    /* setup pci memory regions */

    memory_region_init_alias(&mch->pci_hole, OBJECT(mch), "pci-hole",

                             mch->pci_address_space,

                             mch->below_4g_mem_size,

                             0x100000000ULL - mch->below_4g_mem_size);

    memory_region_add_subregion(mch->system_memory, mch->below_4g_mem_size,

                                &mch->pci_hole);

    pci_hole64_size = (sizeof(hwaddr) == 4 ? 0 :

                       ((uint64_t)1 << 62));

    memory_region_init_alias(&mch->pci_hole_64bit, OBJECT(mch), "pci-hole64",

                             mch->pci_address_space,

                             0x100000000ULL + mch->above_4g_mem_size,

                             pci_hole64_size);

    if (pci_hole64_size) {

        memory_region_add_subregion(mch->system_memory,

                                    0x100000000ULL + mch->above_4g_mem_size,

                                    &mch->pci_hole_64bit);

    }

    /* smram */

    cpu_smm_register(&mch_set_smm, mch);

    memory_region_init_alias(&mch->smram_region, OBJECT(mch), "smram-region",

                             mch->pci_address_space, 0xa0000, 0x20000);

    memory_region_add_subregion_overlap(mch->system_memory, 0xa0000,

                                        &mch->smram_region, 1);

    memory_region_set_enabled(&mch->smram_region, false);

    init_pam(DEVICE(mch), mch->ram_memory, mch->system_memory, mch->pci_address_space,

             &mch->pam_regions[0], PAM_BIOS_BASE, PAM_BIOS_SIZE);

    for (i = 0; i < 12; ++i) {

        init_pam(DEVICE(mch), mch->ram_memory, mch->system_memory, mch->pci_address_space,

                 &mch->pam_regions[i+1], PAM_EXPAN_BASE + i * PAM_EXPAN_SIZE,

                 PAM_EXPAN_SIZE);

    }

    return 0;

}
