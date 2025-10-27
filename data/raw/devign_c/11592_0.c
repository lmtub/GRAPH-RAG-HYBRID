static void sdhci_write_block_to_card(SDHCIState *s)

{

    int index = 0;



    if (s->prnsts & SDHC_SPACE_AVAILABLE) {

        if (s->norintstsen & SDHC_NISEN_WBUFRDY) {

            s->norintsts |= SDHC_NIS_WBUFRDY;

        }

        sdhci_update_irq(s);

        return;

    }



    if (s->trnmod & SDHC_TRNS_BLK_CNT_EN) {

        if (s->blkcnt == 0) {

            return;

        } else {

            s->blkcnt--;

        }

    }



    for (index = 0; index < (s->blksize & 0x0fff); index++) {

        sd_write_data(s->card, s->fifo_buffer[index]);

    }



    /* Next data can be written through BUFFER DATORT register */

    s->prnsts |= SDHC_SPACE_AVAILABLE;



    /* Finish transfer if that was the last block of data */

    if ((s->trnmod & SDHC_TRNS_MULTI) == 0 ||

            ((s->trnmod & SDHC_TRNS_MULTI) &&

            (s->trnmod & SDHC_TRNS_BLK_CNT_EN) && (s->blkcnt == 0))) {

        SDHCI_GET_CLASS(s)->end_data_transfer(s);

    } else if (s->norintstsen & SDHC_NISEN_WBUFRDY) {

        s->norintsts |= SDHC_NIS_WBUFRDY;

    }



    /* Generate Block Gap Event if requested and if not the last block */

    if (s->stopped_state == sdhc_gap_write && (s->trnmod & SDHC_TRNS_MULTI) &&

            s->blkcnt > 0) {

        s->prnsts &= ~SDHC_DOING_WRITE;

        if (s->norintstsen & SDHC_EISEN_BLKGAP) {

            s->norintsts |= SDHC_EIS_BLKGAP;

        }

        SDHCI_GET_CLASS(s)->end_data_transfer(s);

    }



    sdhci_update_irq(s);

}
