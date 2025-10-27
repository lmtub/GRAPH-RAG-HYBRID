static void pxa2xx_descriptor_load(PXA2xxLCDState *s)

{

    PXAFrameDescriptor desc;

    target_phys_addr_t descptr;

    int i;



    for (i = 0; i < PXA_LCDDMA_CHANS; i ++) {

        s->dma_ch[i].source = 0;



        if (!s->dma_ch[i].up)

            continue;



        if (s->dma_ch[i].branch & FBR_BRA) {

            descptr = s->dma_ch[i].branch & FBR_SRCADDR;

            if (s->dma_ch[i].branch & FBR_BINT)

                pxa2xx_dma_bs_set(s, i);

            s->dma_ch[i].branch &= ~FBR_BRA;

        } else

            descptr = s->dma_ch[i].descriptor;



        if (!(descptr >= PXA2XX_SDRAM_BASE && descptr +

                    sizeof(desc) <= PXA2XX_SDRAM_BASE + ram_size))

            continue;



        cpu_physical_memory_read(descptr, (void *)&desc, sizeof(desc));

        s->dma_ch[i].descriptor = tswap32(desc.fdaddr);

        s->dma_ch[i].source = tswap32(desc.fsaddr);

        s->dma_ch[i].id = tswap32(desc.fidr);

        s->dma_ch[i].command = tswap32(desc.ldcmd);

    }

}
