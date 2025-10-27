static void mch_update_smram(MCHPCIState *mch)

{

    PCIDevice *pd = PCI_DEVICE(mch);

    bool h_smrame = (pd->config[MCH_HOST_BRIDGE_ESMRAMC] & MCH_HOST_BRIDGE_ESMRAMC_H_SMRAME);

    uint32_t tseg_size;



    /* implement SMRAM.D_LCK */

    if (pd->config[MCH_HOST_BRIDGE_SMRAM] & MCH_HOST_BRIDGE_SMRAM_D_LCK) {

        pd->config[MCH_HOST_BRIDGE_SMRAM] &= ~MCH_HOST_BRIDGE_SMRAM_D_OPEN;

        pd->wmask[MCH_HOST_BRIDGE_SMRAM] = MCH_HOST_BRIDGE_SMRAM_WMASK_LCK;

        pd->wmask[MCH_HOST_BRIDGE_ESMRAMC] = MCH_HOST_BRIDGE_ESMRAMC_WMASK_LCK;

    }



    memory_region_transaction_begin();



    if (pd->config[MCH_HOST_BRIDGE_SMRAM] & SMRAM_D_OPEN) {

        /* Hide (!) low SMRAM if H_SMRAME = 1 */

        memory_region_set_enabled(&mch->smram_region, h_smrame);

        /* Show high SMRAM if H_SMRAME = 1 */

        memory_region_set_enabled(&mch->open_high_smram, h_smrame);

    } else {

        /* Hide high SMRAM and low SMRAM */

        memory_region_set_enabled(&mch->smram_region, true);

        memory_region_set_enabled(&mch->open_high_smram, false);

    }



    if (pd->config[MCH_HOST_BRIDGE_SMRAM] & SMRAM_G_SMRAME) {

        memory_region_set_enabled(&mch->low_smram, !h_smrame);

        memory_region_set_enabled(&mch->high_smram, h_smrame);

    } else {

        memory_region_set_enabled(&mch->low_smram, false);

        memory_region_set_enabled(&mch->high_smram, false);

    }



    if (pd->config[MCH_HOST_BRIDGE_ESMRAMC] & MCH_HOST_BRIDGE_ESMRAMC_T_EN) {

        switch (pd->config[MCH_HOST_BRIDGE_ESMRAMC] &

                MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_MASK) {

        case MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_1MB:

            tseg_size = 1024 * 1024;

            break;

        case MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_2MB:

            tseg_size = 1024 * 1024 * 2;

            break;

        case MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_8MB:

            tseg_size = 1024 * 1024 * 8;

            break;

        default:

            tseg_size = 0;

            break;

        }

    } else {

        tseg_size = 0;

    }

    memory_region_del_subregion(mch->system_memory, &mch->tseg_blackhole);

    memory_region_set_enabled(&mch->tseg_blackhole, tseg_size);

    memory_region_set_size(&mch->tseg_blackhole, tseg_size);

    memory_region_add_subregion_overlap(mch->system_memory,

                                        mch->below_4g_mem_size - tseg_size,

                                        &mch->tseg_blackhole, 1);



    memory_region_set_enabled(&mch->tseg_window, tseg_size);

    memory_region_set_size(&mch->tseg_window, tseg_size);

    memory_region_set_address(&mch->tseg_window,

                              mch->below_4g_mem_size - tseg_size);

    memory_region_set_alias_offset(&mch->tseg_window,

                                   mch->below_4g_mem_size - tseg_size);



    memory_region_transaction_commit();

}
