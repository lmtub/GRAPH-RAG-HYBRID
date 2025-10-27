static int ahci_populate_sglist(AHCIDevice *ad, QEMUSGList *sglist)

{

    AHCICmdHdr *cmd = ad->cur_cmd;

    uint32_t opts = le32_to_cpu(cmd->opts);

    uint64_t prdt_addr = le64_to_cpu(cmd->tbl_addr) + 0x80;

    int sglist_alloc_hint = opts >> AHCI_CMD_HDR_PRDT_LEN;

    dma_addr_t prdt_len = (sglist_alloc_hint * sizeof(AHCI_SG));

    dma_addr_t real_prdt_len = prdt_len;

    uint8_t *prdt;

    int i;

    int r = 0;



    if (!sglist_alloc_hint) {

        DPRINTF(ad->port_no, "no sg list given by guest: 0x%08x\n", opts);

        return -1;

    }



    /* map PRDT */

    if (!(prdt = dma_memory_map(ad->hba->dma, prdt_addr, &prdt_len,

                                DMA_DIRECTION_TO_DEVICE))){

        DPRINTF(ad->port_no, "map failed\n");

        return -1;

    }



    if (prdt_len < real_prdt_len) {

        DPRINTF(ad->port_no, "mapped less than expected\n");

        r = -1;

        goto out;

    }



    /* Get entries in the PRDT, init a qemu sglist accordingly */

    if (sglist_alloc_hint > 0) {

        AHCI_SG *tbl = (AHCI_SG *)prdt;



        qemu_sglist_init(sglist, sglist_alloc_hint, ad->hba->dma);

        for (i = 0; i < sglist_alloc_hint; i++) {

            /* flags_size is zero-based */

            qemu_sglist_add(sglist, le64_to_cpu(tbl[i].addr),

                            le32_to_cpu(tbl[i].flags_size) + 1);

        }

    }



out:

    dma_memory_unmap(ad->hba->dma, prdt, prdt_len,

                     DMA_DIRECTION_TO_DEVICE, prdt_len);

    return r;

}
