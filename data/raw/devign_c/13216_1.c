static int dnxhd_init_vlc(DNXHDContext *ctx, int cid)

{

    if (cid != ctx->cid) {

        int index;



        if ((index = ff_dnxhd_get_cid_table(cid)) < 0) {

            av_log(ctx->avctx, AV_LOG_ERROR, "unsupported cid %d\n", cid);

            return -1;

        }

        if (ff_dnxhd_cid_table[index].bit_depth != ctx->bit_depth) {

            av_log(ctx->avctx, AV_LOG_ERROR, "bit depth mismatches %d %d\n", ff_dnxhd_cid_table[index].bit_depth, ctx->bit_depth);

            return AVERROR_INVALIDDATA;

        }

        ctx->cid_table = &ff_dnxhd_cid_table[index];



        ff_free_vlc(&ctx->ac_vlc);

        ff_free_vlc(&ctx->dc_vlc);

        ff_free_vlc(&ctx->run_vlc);



        init_vlc(&ctx->ac_vlc, DNXHD_VLC_BITS, 257,

                 ctx->cid_table->ac_bits, 1, 1,

                 ctx->cid_table->ac_codes, 2, 2, 0);

        init_vlc(&ctx->dc_vlc, DNXHD_DC_VLC_BITS, ctx->bit_depth + 4,

                 ctx->cid_table->dc_bits, 1, 1,

                 ctx->cid_table->dc_codes, 1, 1, 0);

        init_vlc(&ctx->run_vlc, DNXHD_VLC_BITS, 62,

                 ctx->cid_table->run_bits, 1, 1,

                 ctx->cid_table->run_codes, 2, 2, 0);



        ff_init_scantable(ctx->dsp.idct_permutation, &ctx->scantable, ff_zigzag_direct);

        ctx->cid = cid;

    }

    return 0;

}
