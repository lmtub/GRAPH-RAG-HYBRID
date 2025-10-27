static int decode_picture_timing(H264Context *h)

{

    if (h->sps.nal_hrd_parameters_present_flag ||

        h->sps.vcl_hrd_parameters_present_flag) {

        h->sei_cpb_removal_delay = get_bits(&h->gb,

                                            h->sps.cpb_removal_delay_length);

        h->sei_dpb_output_delay  = get_bits(&h->gb,

                                            h->sps.dpb_output_delay_length);

    }

    if (h->sps.pic_struct_present_flag) {

        unsigned int i, num_clock_ts;



        h->sei_pic_struct = get_bits(&h->gb, 4);

        h->sei_ct_type    = 0;



        if (h->sei_pic_struct > SEI_PIC_STRUCT_FRAME_TRIPLING)

            return AVERROR_INVALIDDATA;



        num_clock_ts = sei_num_clock_ts_table[h->sei_pic_struct];



        for (i = 0; i < num_clock_ts; i++) {

            if (get_bits(&h->gb, 1)) {                /* clock_timestamp_flag */

                unsigned int full_timestamp_flag;



                h->sei_ct_type |= 1 << get_bits(&h->gb, 2);

                skip_bits(&h->gb, 1);                 /* nuit_field_based_flag */

                skip_bits(&h->gb, 5);                 /* counting_type */

                full_timestamp_flag = get_bits(&h->gb, 1);

                skip_bits(&h->gb, 1);                 /* discontinuity_flag */

                skip_bits(&h->gb, 1);                 /* cnt_dropped_flag */

                skip_bits(&h->gb, 8);                 /* n_frames */

                if (full_timestamp_flag) {

                    skip_bits(&h->gb, 6);             /* seconds_value 0..59 */

                    skip_bits(&h->gb, 6);             /* minutes_value 0..59 */

                    skip_bits(&h->gb, 5);             /* hours_value 0..23 */

                } else {

                    if (get_bits(&h->gb, 1)) {        /* seconds_flag */

                        skip_bits(&h->gb, 6);         /* seconds_value range 0..59 */

                        if (get_bits(&h->gb, 1)) {    /* minutes_flag */

                            skip_bits(&h->gb, 6);     /* minutes_value 0..59 */

                            if (get_bits(&h->gb, 1))  /* hours_flag */

                                skip_bits(&h->gb, 5); /* hours_value 0..23 */

                        }

                    }

                }

                if (h->sps.time_offset_length > 0)

                    skip_bits(&h->gb,

                              h->sps.time_offset_length); /* time_offset */

            }

        }



        if (h->avctx->debug & FF_DEBUG_PICT_INFO)

            av_log(h->avctx, AV_LOG_DEBUG, "ct_type:%X pic_struct:%d\n",

                   h->sei_ct_type, h->sei_pic_struct);

    }

    return 0;

}
