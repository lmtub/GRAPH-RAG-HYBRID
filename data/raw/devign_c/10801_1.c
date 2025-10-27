static int ahci_populate_sglist(AHCIDevice *ad, QEMUSGList *sglist, int offset)

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

    int sum = 0;

    int off_idx = -1;

    int off_pos = -1;

    int tbl_entry_size;

    IDEBus *bus = &ad->port;

    BusState *qbus = BUS(bus);



    if (!sglist_alloc_hint) {

        DPRINTF(ad->port_no, "no sg list given by guest: 0x%08x\n", opts);

        return -1;

    }



    /* map PRDT */

    if (!(prdt = dma_memory_map(ad->hba->as, prdt_addr, &prdt_len,

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

        sum = 0;

        for (i = 0; i < sglist_alloc_hint; i++) {

            /* flags_size is zero-based */

            tbl_entry_size = (le32_to_cpu(tbl[i].flags_size) + 1);

            if (offset <= (sum + tbl_entry_size)) {

                off_idx = i;

                off_pos = offset - sum;

                break;

            }

            sum += tbl_entry_size;

        }

        if ((off_idx == -1) || (off_pos < 0) || (off_pos > tbl_entry_size)) {

            DPRINTF(ad->port_no, "%s: Incorrect offset! "

                            "off_idx: %d, off_pos: %d\n",

                            __func__, off_idx, off_pos);

            r = -1;

            goto out;

        }



        qemu_sglist_init(sglist, qbus->parent, (sglist_alloc_hint - off_idx),

                         ad->hba->as);

        qemu_sglist_add(sglist, le64_to_cpu(tbl[off_idx].addr + off_pos),

                        le32_to_cpu(tbl[off_idx].flags_size) + 1 - off_pos);



        for (i = off_idx + 1; i < sglist_alloc_hint; i++) {

            /* flags_size is zero-based */

            qemu_sglist_add(sglist, le64_to_cpu(tbl[i].addr),

                            le32_to_cpu(tbl[i].flags_size) + 1);

        }

    }



out:

    dma_memory_unmap(ad->hba->as, prdt, prdt_len,

                     DMA_DIRECTION_TO_DEVICE, prdt_len);

    return r;

}
