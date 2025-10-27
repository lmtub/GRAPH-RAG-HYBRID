static uint32_t sdhci_read_dataport(SDHCIState *s, unsigned size)

{

    uint32_t value = 0;

    int i;



    /* first check that a valid data exists in host controller input buffer */

    if ((s->prnsts & SDHC_DATA_AVAILABLE) == 0) {

        ERRPRINT("Trying to read from empty buffer\n");

        return 0;

    }



    for (i = 0; i < size; i++) {

        value |= s->fifo_buffer[s->data_count] << i * 8;

        s->data_count++;

        /* check if we've read all valid data (blksize bytes) from buffer */

        if ((s->data_count) >= (s->blksize & 0x0fff)) {

            DPRINT_L2("All %u bytes of data have been read from input buffer\n",

                    s->data_count);

            s->prnsts &= ~SDHC_DATA_AVAILABLE; /* no more data in a buffer */

            s->data_count = 0;  /* next buff read must start at position [0] */



            if (s->trnmod & SDHC_TRNS_BLK_CNT_EN) {

                s->blkcnt--;

            }



            /* if that was the last block of data */

            if ((s->trnmod & SDHC_TRNS_MULTI) == 0 ||

                ((s->trnmod & SDHC_TRNS_BLK_CNT_EN) && (s->blkcnt == 0)) ||

                 /* stop at gap request */

                (s->stopped_state == sdhc_gap_read &&

                 !(s->prnsts & SDHC_DAT_LINE_ACTIVE))) {

                sdhci_end_transfer(s);

            } else { /* if there are more data, read next block from card */

                sdhci_read_block_from_card(s);

            }

            break;

        }

    }



    return value;

}
