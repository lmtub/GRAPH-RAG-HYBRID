static void intel_hda_set_st_ctl(IntelHDAState *d, const IntelHDAReg *reg, uint32_t old)

{

    bool output = reg->stream >= 4;

    IntelHDAStream *st = d->st + reg->stream;



    if (st->ctl & 0x01) {

        /* reset */

        dprint(d, 1, "st #%d: reset\n", reg->stream);

        st->ctl = 0;

    }

    if ((st->ctl & 0x02) != (old & 0x02)) {

        uint32_t stnr = (st->ctl >> 20) & 0x0f;

        /* run bit flipped */

        if (st->ctl & 0x02) {

            /* start */

            dprint(d, 1, "st #%d: start %d (ring buf %d bytes)\n",

                   reg->stream, stnr, st->cbl);

            intel_hda_parse_bdl(d, st);

            intel_hda_notify_codecs(d, stnr, true, output);

        } else {

            /* stop */

            dprint(d, 1, "st #%d: stop %d\n", reg->stream, stnr);

            intel_hda_notify_codecs(d, stnr, false, output);

        }

    }

    intel_hda_update_irq(d);

}
