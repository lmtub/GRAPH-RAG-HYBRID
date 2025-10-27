static void mch_realize(PCIDevice *d, Error **errp)

{

    int i;

    MCHPCIState *mch = MCH_PCI_DEVICE(d);



    /* setup pci memory mapping */

    pc_pci_as_mapping_init(OBJECT(mch), mch->system_memory,

                           mch->pci_address_space);



    /* smram */

    cpu_smm_register(&mch_set_smm, mch);

    memory_region_init_alias(&mch->smram_region, OBJECT(mch), "smram-region",

                             mch->pci_address_space, 0xa0000, 0x20000);

    memory_region_add_subregion_overlap(mch->system_memory, 0xa0000,

                                        &mch->smram_region, 1);

    memory_region_set_enabled(&mch->smram_region, false);

    init_pam(DEVICE(mch), mch->ram_memory, mch->system_memory,

             mch->pci_address_space, &mch->pam_regions[0],

             PAM_BIOS_BASE, PAM_BIOS_SIZE);

    for (i = 0; i < 12; ++i) {

        init_pam(DEVICE(mch), mch->ram_memory, mch->system_memory,

                 mch->pci_address_space, &mch->pam_regions[i+1],

                 PAM_EXPAN_BASE + i * PAM_EXPAN_SIZE, PAM_EXPAN_SIZE);

    }

    /* Intel IOMMU (VT-d) */

    if (qemu_opt_get_bool(qemu_get_machine_opts(), "iommu", false)) {

        mch_init_dmar(mch);

    }

}
