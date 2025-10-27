static void set_downmix_coeffs(AC3DecodeContext *s)

{

    int i;

    float cmix = gain_levels[s->center_mix_level];

    float smix = gain_levels[s->surround_mix_level];



    for(i=0; i<s->fbw_channels; i++) {

        s->downmix_coeffs[i][0] = gain_levels[ac3_default_coeffs[s->channel_mode][i][0]];

        s->downmix_coeffs[i][1] = gain_levels[ac3_default_coeffs[s->channel_mode][i][1]];

    }

    if(s->channel_mode > 1 && s->channel_mode & 1) {

        s->downmix_coeffs[1][0] = s->downmix_coeffs[1][1] = cmix;

    }

    if(s->channel_mode == AC3_CHMODE_2F1R || s->channel_mode == AC3_CHMODE_3F1R) {

        int nf = s->channel_mode - 2;

        s->downmix_coeffs[nf][0] = s->downmix_coeffs[nf][1] = smix * LEVEL_MINUS_3DB;

    }

    if(s->channel_mode == AC3_CHMODE_2F2R || s->channel_mode == AC3_CHMODE_3F2R) {

        int nf = s->channel_mode - 4;

        s->downmix_coeffs[nf][0] = s->downmix_coeffs[nf+1][1] = smix;

    }



    s->downmix_coeff_sum[0] = s->downmix_coeff_sum[1] = 0.0f;

    for(i=0; i<s->fbw_channels; i++) {

        s->downmix_coeff_sum[0] += s->downmix_coeffs[i][0];

        s->downmix_coeff_sum[1] += s->downmix_coeffs[i][1];

    }

}
