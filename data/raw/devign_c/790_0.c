static void rx_init_frame(eTSEC *etsec, const uint8_t *buf, size_t size)

{

    uint32_t fcb_size = 0;

    uint8_t  prsdep   = (etsec->regs[RCTRL].value >> RCTRL_PRSDEP_OFFSET)

        & RCTRL_PRSDEP_MASK;



    if (prsdep != 0) {

        /* Prepend FCB (FCB size + RCTRL[PAL]) */

        fcb_size = 8 + ((etsec->regs[RCTRL].value >> 16) & 0x1F);



        etsec->rx_fcb_size = fcb_size;



        /* TODO: fill_FCB(etsec); */

        memset(etsec->rx_fcb, 0x0, sizeof(etsec->rx_fcb));



    } else {

        etsec->rx_fcb_size = 0;

    }



    if (etsec->rx_buffer != NULL) {

        g_free(etsec->rx_buffer);

    }



    /* Do not copy the frame for now */

    etsec->rx_buffer     = (uint8_t *)buf;

    etsec->rx_buffer_len = size;



    /* CRC padding (We don't have to compute the CRC) */

    etsec->rx_padding = 4;



    etsec->rx_first_in_frame = 1;

    etsec->rx_remaining_data = etsec->rx_buffer_len;

    RING_DEBUG("%s: rx_buffer_len:%u rx_padding+crc:%u\n", __func__,

               etsec->rx_buffer_len, etsec->rx_padding);

}
