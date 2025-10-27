static void encode_scale_factors(AVCodecContext *avctx, AACEncContext *s,

                                 SingleChannelElement *sce)

{

    int off = sce->sf_idx[0], diff;

    int i, w;



    for (w = 0; w < sce->ics.num_windows; w += sce->ics.group_len[w]) {

        for (i = 0; i < sce->ics.max_sfb; i++) {

            if (!sce->zeroes[w*16 + i]) {

                diff = sce->sf_idx[w*16 + i] - off + SCALE_DIFF_ZERO;

                if (diff < 0 || diff > 120)

                    av_log(avctx, AV_LOG_ERROR, "Scalefactor difference is too big to be coded\n");

                off = sce->sf_idx[w*16 + i];

                put_bits(&s->pb, ff_aac_scalefactor_bits[diff], ff_aac_scalefactor_code[diff]);

            }

        }

    }

}
