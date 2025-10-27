static int sbr_make_f_master(AACContext *ac, SpectralBandReplication *sbr,

                             SpectrumParameters *spectrum)

{

    unsigned int temp, max_qmf_subbands = 0;

    unsigned int start_min, stop_min;

    int k;

    const int8_t *sbr_offset_ptr;

    int16_t stop_dk[13];



    if (sbr->sample_rate < 32000) {

        temp = 3000;

    } else if (sbr->sample_rate < 64000) {

        temp = 4000;

    } else

        temp = 5000;



    start_min = ((temp << 7) + (sbr->sample_rate >> 1)) / sbr->sample_rate;

    stop_min  = ((temp << 8) + (sbr->sample_rate >> 1)) / sbr->sample_rate;



    switch (sbr->sample_rate) {

    case 16000:

        sbr_offset_ptr = sbr_offset[0];

        break;

    case 22050:

        sbr_offset_ptr = sbr_offset[1];

        break;

    case 24000:

        sbr_offset_ptr = sbr_offset[2];

        break;

    case 32000:

        sbr_offset_ptr = sbr_offset[3];

        break;

    case 44100: case 48000: case 64000:

        sbr_offset_ptr = sbr_offset[4];

        break;

    case 88200: case 96000: case 128000: case 176400: case 192000:

        sbr_offset_ptr = sbr_offset[5];

        break;

    default:

        av_log(ac->avctx, AV_LOG_ERROR,

               "Unsupported sample rate for SBR: %d\n", sbr->sample_rate);

        return -1;

    }



    sbr->k[0] = start_min + sbr_offset_ptr[spectrum->bs_start_freq];



    if (spectrum->bs_stop_freq < 14) {

        sbr->k[2] = stop_min;

        make_bands(stop_dk, stop_min, 64, 13);

        qsort(stop_dk, 13, sizeof(stop_dk[0]), qsort_comparison_function_int16);

        for (k = 0; k < spectrum->bs_stop_freq; k++)

            sbr->k[2] += stop_dk[k];

    } else if (spectrum->bs_stop_freq == 14) {

        sbr->k[2] = 2*sbr->k[0];

    } else if (spectrum->bs_stop_freq == 15) {

        sbr->k[2] = 3*sbr->k[0];

    } else {

        av_log(ac->avctx, AV_LOG_ERROR,

               "Invalid bs_stop_freq: %d\n", spectrum->bs_stop_freq);

        return -1;

    }

    sbr->k[2] = FFMIN(64, sbr->k[2]);



    // Requirements (14496-3 sp04 p205)

    if (sbr->sample_rate <= 32000) {

        max_qmf_subbands = 48;

    } else if (sbr->sample_rate == 44100) {

        max_qmf_subbands = 35;

    } else if (sbr->sample_rate >= 48000)

        max_qmf_subbands = 32;



    if (sbr->k[2] - sbr->k[0] > max_qmf_subbands) {

        av_log(ac->avctx, AV_LOG_ERROR,

               "Invalid bitstream, too many QMF subbands: %d\n", sbr->k[2] - sbr->k[0]);

        return -1;

    }



    if (!spectrum->bs_freq_scale) {

        int dk, k2diff;



        dk = spectrum->bs_alter_scale + 1;

        sbr->n_master = ((sbr->k[2] - sbr->k[0] + (dk&2)) >> dk) << 1;

        if (check_n_master(ac->avctx, sbr->n_master, sbr->spectrum_params.bs_xover_band))

            return -1;



        for (k = 1; k <= sbr->n_master; k++)

            sbr->f_master[k] = dk;



        k2diff = sbr->k[2] - sbr->k[0] - sbr->n_master * dk;

        if (k2diff < 0) {

            sbr->f_master[1]--;

            sbr->f_master[2]-= (k2diff < -1);

        } else if (k2diff) {

            sbr->f_master[sbr->n_master]++;

        }



        sbr->f_master[0] = sbr->k[0];

        for (k = 1; k <= sbr->n_master; k++)

            sbr->f_master[k] += sbr->f_master[k - 1];



    } else {

        int half_bands = 7 - spectrum->bs_freq_scale;      // bs_freq_scale  = {1,2,3}

        int two_regions, num_bands_0;

        int vdk0_max, vdk1_min;

        int16_t vk0[49];



        if (49 * sbr->k[2] > 110 * sbr->k[0]) {

            two_regions = 1;

            sbr->k[1] = 2 * sbr->k[0];

        } else {

            two_regions = 0;

            sbr->k[1] = sbr->k[2];

        }



        num_bands_0 = lrintf(half_bands * log2f(sbr->k[1] / (float)sbr->k[0])) * 2;



        if (num_bands_0 <= 0) { // Requirements (14496-3 sp04 p205)

            av_log(ac->avctx, AV_LOG_ERROR, "Invalid num_bands_0: %d\n", num_bands_0);

            return -1;

        }



        vk0[0] = 0;



        make_bands(vk0+1, sbr->k[0], sbr->k[1], num_bands_0);



        qsort(vk0 + 1, num_bands_0, sizeof(vk0[1]), qsort_comparison_function_int16);

        vdk0_max = vk0[num_bands_0];



        vk0[0] = sbr->k[0];

        for (k = 1; k <= num_bands_0; k++) {

            if (vk0[k] <= 0) { // Requirements (14496-3 sp04 p205)

                av_log(ac->avctx, AV_LOG_ERROR, "Invalid vDk0[%d]: %d\n", k, vk0[k]);

                return -1;

            }

            vk0[k] += vk0[k-1];

        }



        if (two_regions) {

            int16_t vk1[49];

            float invwarp = spectrum->bs_alter_scale ? 0.76923076923076923077f

                                                     : 1.0f; // bs_alter_scale = {0,1}

            int num_bands_1 = lrintf(half_bands * invwarp *

                                     log2f(sbr->k[2] / (float)sbr->k[1])) * 2;



            make_bands(vk1+1, sbr->k[1], sbr->k[2], num_bands_1);



            vdk1_min = array_min_int16(vk1 + 1, num_bands_1);



            if (vdk1_min < vdk0_max) {

                int change;

                qsort(vk1 + 1, num_bands_1, sizeof(vk1[1]), qsort_comparison_function_int16);

                change = FFMIN(vdk0_max - vk1[1], (vk1[num_bands_1] - vk1[1]) >> 1);

                vk1[1]           += change;

                vk1[num_bands_1] -= change;

            }



            qsort(vk1 + 1, num_bands_1, sizeof(vk1[1]), qsort_comparison_function_int16);



            vk1[0] = sbr->k[1];

            for (k = 1; k <= num_bands_1; k++) {

                if (vk1[k] <= 0) { // Requirements (14496-3 sp04 p205)

                    av_log(ac->avctx, AV_LOG_ERROR, "Invalid vDk1[%d]: %d\n", k, vk1[k]);

                    return -1;

                }

                vk1[k] += vk1[k-1];

            }



            sbr->n_master = num_bands_0 + num_bands_1;

            if (check_n_master(ac->avctx, sbr->n_master, sbr->spectrum_params.bs_xover_band))

                return -1;

            memcpy(&sbr->f_master[0],               vk0,

                   (num_bands_0 + 1) * sizeof(sbr->f_master[0]));

            memcpy(&sbr->f_master[num_bands_0 + 1], vk1 + 1,

                    num_bands_1      * sizeof(sbr->f_master[0]));



        } else {

            sbr->n_master = num_bands_0;

            if (check_n_master(ac->avctx, sbr->n_master, sbr->spectrum_params.bs_xover_band))

                return -1;

            memcpy(sbr->f_master, vk0, (num_bands_0 + 1) * sizeof(sbr->f_master[0]));

        }

    }



    return 0;

}
