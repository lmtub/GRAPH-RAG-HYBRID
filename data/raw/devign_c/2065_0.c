static void mch_realize(PCIDevice *d, Error **errp)

{

    int i;

    MCHPCIState *mch = MCH_PCI_DEVICE(d);



    /* setup pci memory mapping */

    pc_pci_as_mapping_init(OBJECT(mch), mch->system_memory,

                           mch->pci_address_space);



    /* if *disabled* show SMRAM to all CPUs */

    memory_region_init_alias(&mch->smram_region, OBJECT(mch), "smram-region",

                             mch->pci_address_space, 0xa0000, 0x20000);

    memory_region_add_subregion_overlap(mch->system_memory, 0xa0000,

                                        &mch->smram_region, 1);

    memory_region_set_enabled(&mch->smram_region, true);



    memory_region_init_alias(&mch->open_high_smram, OBJECT(mch), "smram-open-high",

                             mch->ram_memory, 0xa0000, 0x20000);

    memory_region_add_subregion_overlap(mch->system_memory, 0xfeda0000,

                                        &mch->open_high_smram, 1);

    memory_region_set_enabled(&mch->open_high_smram, false);



    /* smram, as seen by SMM CPUs */

    memory_region_init(&mch->smram, OBJECT(mch), "smram", 1ull << 32);

    memory_region_set_enabled(&mch->smram, true);

    memory_region_init_alias(&mch->low_smram, OBJECT(mch), "smram-low",

                             mch->ram_memory, 0xa0000, 0x20000);

    memory_region_set_enabled(&mch->low_smram, true);

    memory_region_add_subregion(&mch->smram, 0xa0000, &mch->low_smram);

    memory_region_init_alias(&mch->high_smram, OBJECT(mch), "smram-high",

                             mch->ram_memory, 0xa0000, 0x20000);

    memory_region_set_enabled(&mch->high_smram, true);

    memory_region_add_subregion(&mch->smram, 0xfeda0000, &mch->high_smram);



    memory_region_init_io(&mch->tseg_blackhole, OBJECT(mch),

                          &tseg_blackhole_ops, NULL,

                          "tseg-blackhole", 0);

    memory_region_set_enabled(&mch->tseg_blackhole, false);

    memory_region_add_subregion_overlap(mch->system_memory,

                                        mch->below_4g_mem_size,

                                        &mch->tseg_blackhole, 1);



    memory_region_init_alias(&mch->tseg_window, OBJECT(mch), "tseg-window",

                             mch->ram_memory, mch->below_4g_mem_size, 0);

    memory_region_set_enabled(&mch->tseg_window, false);

    memory_region_add_subregion(&mch->smram, mch->below_4g_mem_size,

                                &mch->tseg_window);

    object_property_add_const_link(qdev_get_machine(), "smram",

                                   OBJECT(&mch->smram), &error_abort);



    init_pam(DEVICE(mch), mch->ram_memory, mch->system_memory,

             mch->pci_address_space, &mch->pam_regions[0],

             PAM_BIOS_BASE, PAM_BIOS_SIZE);

    for (i = 0; i < 12; ++i) {

        init_pam(DEVICE(mch), mch->ram_memory, mch->system_memory,

                 mch->pci_address_space, &mch->pam_regions[i+1],

                 PAM_EXPAN_BASE + i * PAM_EXPAN_SIZE, PAM_EXPAN_SIZE);

    }

    /* Intel IOMMU (VT-d) */

    if (machine_iommu(current_machine)) {

        mch_init_dmar(mch);

    }

}
