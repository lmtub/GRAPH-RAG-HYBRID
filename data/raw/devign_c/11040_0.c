static int ac3_parse_bsi(AC3DecodeContext *ctx)

{

    ac3_bsi *bsi = &ctx->bsi;

    uint32_t *flags = &bsi->flags;

    GetBitContext *gb = &ctx->gb;



    *flags = 0;

    bsi->cmixlev = 0;

    bsi->surmixlev = 0;

    bsi->dsurmod = 0;



    bsi->bsid = get_bits(gb, 5);

    if (bsi->bsid > 0x08)

        return -1;

    bsi->bsmod = get_bits(gb, 3);

    bsi->acmod = get_bits(gb, 3);

    if (bsi->acmod & 0x01 && bsi->acmod != 0x01)

        bsi->cmixlev = get_bits(gb, 2);

    if (bsi->acmod & 0x04)

        bsi->surmixlev = get_bits(gb, 2);

    if (bsi->acmod == 0x02)

        bsi->dsurmod = get_bits(gb, 2);

    if (get_bits(gb, 1))

        *flags |= AC3_BSI_LFEON;

    bsi->dialnorm = get_bits(gb, 5);

    if (get_bits(gb, 1)) {

        *flags |= AC3_BSI_COMPRE;

        bsi->compr = get_bits(gb, 5);

    }

    if (get_bits(gb, 1)) {

        *flags |= AC3_BSI_LANGCODE;

        bsi->langcod = get_bits(gb, 8);

    }

    if (get_bits(gb, 1)) {

        *flags |= AC3_BSI_AUDPRODIE;

        bsi->mixlevel = get_bits(gb, 5);

        bsi->roomtyp = get_bits(gb, 2);

    }

    if (bsi->acmod == 0x00) {

        bsi->dialnorm2 = get_bits(gb, 5);

        if (get_bits(gb, 1)) {

            *flags |= AC3_BSI_COMPR2E;

            bsi->compr2 = get_bits(gb, 5);

        }

        if (get_bits(gb, 1)) {

            *flags |= AC3_BSI_LANGCOD2E;

            bsi->langcod2 = get_bits(gb, 8);

        }

        if (get_bits(gb, 1)) {

            *flags |= AC3_BSI_AUDPRODIE;

            bsi->mixlevel2 = get_bits(gb, 5);

            bsi->roomtyp2 = get_bits(gb, 2);

        }

    }

    if (get_bits(gb, 1))

        *flags |= AC3_BSI_COPYRIGHTB;

    if (get_bits(gb, 1))

        *flags |= AC3_BSI_ORIGBS;

    if (get_bits(gb, 1)) {

        *flags |= AC3_BSI_TIMECOD1E;

        bsi->timecod1 = get_bits(gb, 14);

    }

    if (get_bits(gb, 1)) {

        *flags |= AC3_BSI_TIMECOD2E;

        bsi->timecod2 = get_bits(gb, 14);

    }

    if (get_bits(gb, 1)) {

        *flags |= AC3_BSI_ADDBSIE;

        bsi->addbsil = get_bits(gb, 6);

        do {

            get_bits(gb, 8);

        } while (bsi->addbsil--);

    }



    bsi->nfchans = nfchans_tbl[bsi->acmod];



    return 0;

}
