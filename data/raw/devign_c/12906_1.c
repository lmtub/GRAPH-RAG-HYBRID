static void spapr_pci_pre_save(void *opaque)

{

    sPAPRPHBState *sphb = opaque;

    GHashTableIter iter;

    gpointer key, value;

    int i;



    g_free(sphb->msi_devs);

    sphb->msi_devs = NULL;

    sphb->msi_devs_num = g_hash_table_size(sphb->msi);

    if (!sphb->msi_devs_num) {

        return;

    }

    sphb->msi_devs = g_malloc(sphb->msi_devs_num * sizeof(spapr_pci_msi_mig));



    g_hash_table_iter_init(&iter, sphb->msi);

    for (i = 0; g_hash_table_iter_next(&iter, &key, &value); ++i) {

        sphb->msi_devs[i].key = *(uint32_t *) key;

        sphb->msi_devs[i].value = *(spapr_pci_msi *) value;

    }



    if (sphb->pre_2_8_migration) {

        sphb->mig_liobn = sphb->dma_liobn[0];

        sphb->mig_mem_win_addr = sphb->mem_win_addr;

        sphb->mig_mem_win_size = sphb->mem_win_size;

        sphb->mig_io_win_addr = sphb->io_win_addr;

        sphb->mig_io_win_size = sphb->io_win_size;



        if ((sphb->mem64_win_size != 0)

            && (sphb->mem64_win_addr

                == (sphb->mem_win_addr + sphb->mem_win_size))) {

            sphb->mig_mem_win_size += sphb->mem64_win_size;

        }

    }

}
