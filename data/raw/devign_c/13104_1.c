static int dca_subframe_footer(DCAContext *s, int base_channel)

{

    int in, out, aux_data_count, aux_data_end, reserved;

    uint32_t nsyncaux;



    /*

     * Unpack optional information

     */



    /* presumably optional information only appears in the core? */

    if (!base_channel) {

        if (s->timestamp)

            skip_bits_long(&s->gb, 32);



        if (s->aux_data) {

            aux_data_count = get_bits(&s->gb, 6);



            // align (32-bit)

            skip_bits_long(&s->gb, (-get_bits_count(&s->gb)) & 31);



            aux_data_end = 8 * aux_data_count + get_bits_count(&s->gb);



            if ((nsyncaux = get_bits_long(&s->gb, 32)) != DCA_NSYNCAUX) {

                av_log(s->avctx, AV_LOG_ERROR, "nSYNCAUX mismatch %#"PRIx32"\n",

                       nsyncaux);

                return AVERROR_INVALIDDATA;

            }



            if (get_bits1(&s->gb)) { // bAUXTimeStampFlag

                avpriv_request_sample(s->avctx,

                                      "Auxiliary Decode Time Stamp Flag");

                // align (4-bit)

                skip_bits(&s->gb, (-get_bits_count(&s->gb)) & 4);

                // 44 bits: nMSByte (8), nMarker (4), nLSByte (28), nMarker (4)

                skip_bits_long(&s->gb, 44);

            }



            if ((s->core_downmix = get_bits1(&s->gb))) {

                int am = get_bits(&s->gb, 3);

                switch (am) {

                case 0:

                    s->core_downmix_amode = DCA_MONO;

                    break;

                case 1:

                    s->core_downmix_amode = DCA_STEREO;

                    break;

                case 2:

                    s->core_downmix_amode = DCA_STEREO_TOTAL;

                    break;

                case 3:

                    s->core_downmix_amode = DCA_3F;

                    break;

                case 4:

                    s->core_downmix_amode = DCA_2F1R;

                    break;

                case 5:

                    s->core_downmix_amode = DCA_2F2R;

                    break;

                case 6:

                    s->core_downmix_amode = DCA_3F1R;

                    break;

                default:

                    av_log(s->avctx, AV_LOG_ERROR,

                           "Invalid mode %d for embedded downmix coefficients\n",

                           am);

                    return AVERROR_INVALIDDATA;

                }

                for (out = 0; out < ff_dca_channels[s->core_downmix_amode]; out++) {

                    for (in = 0; in < s->audio_header.prim_channels + !!s->lfe; in++) {

                        uint16_t tmp = get_bits(&s->gb, 9);

                        if ((tmp & 0xFF) > 241) {

                            av_log(s->avctx, AV_LOG_ERROR,

                                   "Invalid downmix coefficient code %"PRIu16"\n",

                                   tmp);

                            return AVERROR_INVALIDDATA;

                        }

                        s->core_downmix_codes[in][out] = tmp;

                    }

                }

            }



            align_get_bits(&s->gb); // byte align

            skip_bits(&s->gb, 16);  // nAUXCRC16



            // additional data (reserved, cf. ETSI TS 102 114 V1.4.1)

            if ((reserved = (aux_data_end - get_bits_count(&s->gb))) < 0) {

                av_log(s->avctx, AV_LOG_ERROR,

                       "Overread auxiliary data by %d bits\n", -reserved);

                return AVERROR_INVALIDDATA;

            } else if (reserved) {

                avpriv_request_sample(s->avctx,

                                      "Core auxiliary data reserved content");

                skip_bits_long(&s->gb, reserved);

            }

        }



        if (s->crc_present && s->dynrange)

            get_bits(&s->gb, 16);

    }



    return 0;

}
