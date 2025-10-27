static void pxa2xx_update_display(void *opaque)

{

    struct pxa2xx_lcdc_s *s = (struct pxa2xx_lcdc_s *) opaque;

    target_phys_addr_t fbptr;

    int miny, maxy;

    int ch;

    if (!(s->control[0] & LCCR0_ENB))

        return;



    pxa2xx_descriptor_load(s);



    pxa2xx_lcdc_resize(s);

    miny = s->yres;

    maxy = 0;

    s->transp = s->dma_ch[2].up || s->dma_ch[3].up;

    /* Note: With overlay planes the order depends on LCCR0 bit 25.  */

    for (ch = 0; ch < PXA_LCDDMA_CHANS; ch ++)

        if (s->dma_ch[ch].up) {

            if (!s->dma_ch[ch].source) {

                pxa2xx_dma_ber_set(s, ch);

                continue;

            }

            fbptr = s->dma_ch[ch].source;

            if (!(fbptr >= PXA2XX_SDRAM_BASE &&

                    fbptr <= PXA2XX_SDRAM_BASE + phys_ram_size)) {

                pxa2xx_dma_ber_set(s, ch);

                continue;

            }



            if (s->dma_ch[ch].command & LDCMD_PAL) {

                cpu_physical_memory_read(fbptr, s->dma_ch[ch].pbuffer,

                    MAX(LDCMD_LENGTH(s->dma_ch[ch].command),

                        sizeof(s->dma_ch[ch].pbuffer)));

                pxa2xx_palette_parse(s, ch, s->bpp);

            } else {

                /* Do we need to reparse palette */

                if (LCCR4_PALFOR(s->control[4]) != s->pal_for)

                    pxa2xx_palette_parse(s, ch, s->bpp);



                /* ACK frame start */

                pxa2xx_dma_sof_set(s, ch);



                s->dma_ch[ch].redraw(s, fbptr, &miny, &maxy);

                s->invalidated = 0;



                /* ACK frame completed */

                pxa2xx_dma_eof_set(s, ch);

            }

        }



    if (s->control[0] & LCCR0_DIS) {

        /* ACK last frame completed */

        s->control[0] &= ~LCCR0_ENB;

        s->status[0] |= LCSR0_LDD;

    }



    if (miny >= 0) {

        if (s->orientation)

            dpy_update(s->ds, miny, 0, maxy, s->xres);

        else

            dpy_update(s->ds, 0, miny, s->xres, maxy);

    }

    pxa2xx_lcdc_int_update(s);



    qemu_irq_raise(s->vsync_cb);

}
