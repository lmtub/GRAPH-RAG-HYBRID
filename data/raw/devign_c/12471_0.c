static void intel_hda_parse_bdl(IntelHDAState *d, IntelHDAStream *st)

{

    target_phys_addr_t addr;

    uint8_t buf[16];

    uint32_t i;



    addr = intel_hda_addr(st->bdlp_lbase, st->bdlp_ubase);

    st->bentries = st->lvi +1;

    g_free(st->bpl);

    st->bpl = g_malloc(sizeof(bpl) * st->bentries);

    for (i = 0; i < st->bentries; i++, addr += 16) {

        pci_dma_read(&d->pci, addr, buf, 16);

        st->bpl[i].addr  = le64_to_cpu(*(uint64_t *)buf);

        st->bpl[i].len   = le32_to_cpu(*(uint32_t *)(buf + 8));

        st->bpl[i].flags = le32_to_cpu(*(uint32_t *)(buf + 12));

        dprint(d, 1, "bdl/%d: 0x%" PRIx64 " +0x%x, 0x%x\n",

               i, st->bpl[i].addr, st->bpl[i].len, st->bpl[i].flags);

    }



    st->bsize = st->cbl;

    st->lpib  = 0;

    st->be    = 0;

    st->bp    = 0;

}
