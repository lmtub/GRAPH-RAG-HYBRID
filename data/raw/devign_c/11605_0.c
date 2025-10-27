static int dca_exss_parse_asset_header(DCAContext *s)

{

    int header_pos = get_bits_count(&s->gb);

    int header_size;

    int channels;

    int embedded_stereo = 0;

    int embedded_6ch    = 0;

    int drc_code_present;

    int extensions_mask;

    int i, j;



    if (get_bits_left(&s->gb) < 16)

        return -1;



    /* We will parse just enough to get to the extensions bitmask with which

     * we can set the profile value. */



    header_size = get_bits(&s->gb, 9) + 1;

    skip_bits(&s->gb, 3); // asset index



    if (s->static_fields) {

        if (get_bits1(&s->gb))

            skip_bits(&s->gb, 4); // asset type descriptor

        if (get_bits1(&s->gb))

            skip_bits_long(&s->gb, 24); // language descriptor



        if (get_bits1(&s->gb)) {

            /* How can one fit 1024 bytes of text here if the maximum value

             * for the asset header size field above was 512 bytes? */

            int text_length = get_bits(&s->gb, 10) + 1;

            if (get_bits_left(&s->gb) < text_length * 8)

                return -1;

            skip_bits_long(&s->gb, text_length * 8); // info text

        }



        skip_bits(&s->gb, 5); // bit resolution - 1

        skip_bits(&s->gb, 4); // max sample rate code

        channels = get_bits(&s->gb, 8) + 1;



        if (get_bits1(&s->gb)) { // 1-to-1 channels to speakers

            int spkr_remap_sets;

            int spkr_mask_size = 16;

            int num_spkrs[7];



            if (channels > 2)

                embedded_stereo = get_bits1(&s->gb);

            if (channels > 6)

                embedded_6ch = get_bits1(&s->gb);



            if (get_bits1(&s->gb)) {

                spkr_mask_size = (get_bits(&s->gb, 2) + 1) << 2;

                skip_bits(&s->gb, spkr_mask_size); // spkr activity mask

            }



            spkr_remap_sets = get_bits(&s->gb, 3);



            for (i = 0; i < spkr_remap_sets; i++) {

                /* std layout mask for each remap set */

                num_spkrs[i] = dca_exss_mask2count(get_bits(&s->gb, spkr_mask_size));

            }



            for (i = 0; i < spkr_remap_sets; i++) {

                int num_dec_ch_remaps = get_bits(&s->gb, 5) + 1;

                if (get_bits_left(&s->gb) < 0)

                    return -1;



                for (j = 0; j < num_spkrs[i]; j++) {

                    int remap_dec_ch_mask = get_bits_long(&s->gb, num_dec_ch_remaps);

                    int num_dec_ch = av_popcount(remap_dec_ch_mask);

                    skip_bits_long(&s->gb, num_dec_ch * 5); // remap codes

                }

            }

        } else {

            skip_bits(&s->gb, 3); // representation type

        }

    }



    drc_code_present = get_bits1(&s->gb);

    if (drc_code_present)

        get_bits(&s->gb, 8); // drc code



    if (get_bits1(&s->gb))

        skip_bits(&s->gb, 5); // dialog normalization code



    if (drc_code_present && embedded_stereo)

        get_bits(&s->gb, 8); // drc stereo code



    if (s->mix_metadata && get_bits1(&s->gb)) {

        skip_bits(&s->gb, 1); // external mix

        skip_bits(&s->gb, 6); // post mix gain code



        if (get_bits(&s->gb, 2) != 3) // mixer drc code

            skip_bits(&s->gb, 3); // drc limit

        else

            skip_bits(&s->gb, 8); // custom drc code



        if (get_bits1(&s->gb)) // channel specific scaling

            for (i = 0; i < s->num_mix_configs; i++)

                skip_bits_long(&s->gb, s->mix_config_num_ch[i] * 6); // scale codes

        else

            skip_bits_long(&s->gb, s->num_mix_configs * 6); // scale codes



        for (i = 0; i < s->num_mix_configs; i++) {

            if (get_bits_left(&s->gb) < 0)

                return -1;

            dca_exss_skip_mix_coeffs(&s->gb, channels, s->mix_config_num_ch[i]);

            if (embedded_6ch)

                dca_exss_skip_mix_coeffs(&s->gb, 6, s->mix_config_num_ch[i]);

            if (embedded_stereo)

                dca_exss_skip_mix_coeffs(&s->gb, 2, s->mix_config_num_ch[i]);

        }

    }



    switch (get_bits(&s->gb, 2)) {

    case 0:

        extensions_mask = get_bits(&s->gb, 12);

        break;

    case 1:

        extensions_mask = DCA_EXT_EXSS_XLL;

        break;

    case 2:

        extensions_mask = DCA_EXT_EXSS_LBR;

        break;

    case 3:

        extensions_mask = 0; /* aux coding */

        break;

    }



    /* not parsed further, we were only interested in the extensions mask */



    if (get_bits_left(&s->gb) < 0)

        return -1;



    if (get_bits_count(&s->gb) - header_pos > header_size * 8) {

        av_log(s->avctx, AV_LOG_WARNING, "Asset header size mismatch.\n");

        return -1;

    }

    skip_bits_long(&s->gb, header_pos + header_size * 8 - get_bits_count(&s->gb));



    if (extensions_mask & DCA_EXT_EXSS_XLL)

        s->profile = FF_PROFILE_DTS_HD_MA;

    else if (extensions_mask & (DCA_EXT_EXSS_XBR | DCA_EXT_EXSS_X96 |

                                DCA_EXT_EXSS_XXCH))

        s->profile = FF_PROFILE_DTS_HD_HRA;



    if (!(extensions_mask & DCA_EXT_CORE))

        av_log(s->avctx, AV_LOG_WARNING, "DTS core detection mismatch.\n");

    if ((extensions_mask & DCA_CORE_EXTS) != s->core_ext_mask)

        av_log(s->avctx, AV_LOG_WARNING,

               "DTS extensions detection mismatch (%d, %d)\n",

               extensions_mask & DCA_CORE_EXTS, s->core_ext_mask);



    return 0;

}
