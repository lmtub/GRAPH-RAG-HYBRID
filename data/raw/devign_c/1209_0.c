int ff_h264_decode_ref_pic_marking(H264Context *h, GetBitContext *gb,

                                   int first_slice)

{

    int i, ret;

    MMCO mmco_temp[MAX_MMCO_COUNT], *mmco = mmco_temp;

    int mmco_index = 0;



    if (h->nal_unit_type == NAL_IDR_SLICE) { // FIXME fields

        skip_bits1(gb); // broken_link

        if (get_bits1(gb)) {

            mmco[0].opcode   = MMCO_LONG;

            mmco[0].long_arg = 0;

            mmco_index       = 1;

        }

    } else {

        if (get_bits1(gb)) { // adaptive_ref_pic_marking_mode_flag

            for (i = 0; i < MAX_MMCO_COUNT; i++) {

                MMCOOpcode opcode = get_ue_golomb_31(gb);



                mmco[i].opcode = opcode;

                if (opcode == MMCO_SHORT2UNUSED || opcode == MMCO_SHORT2LONG) {

                    mmco[i].short_pic_num =

                        (h->curr_pic_num - get_ue_golomb(gb) - 1) &

                            (h->max_pic_num - 1);

#if 0

                    if (mmco[i].short_pic_num >= h->short_ref_count ||

                        !h->short_ref[mmco[i].short_pic_num]) {

                        av_log(s->avctx, AV_LOG_ERROR,

                               "illegal short ref in memory management control "

                               "operation %d\n", mmco);

                        return -1;

                    }

#endif

                }

                if (opcode == MMCO_SHORT2LONG || opcode == MMCO_LONG2UNUSED ||

                    opcode == MMCO_LONG || opcode == MMCO_SET_MAX_LONG) {

                    unsigned int long_arg = get_ue_golomb_31(gb);

                    if (long_arg >= 32 ||

                        (long_arg >= 16 && !(opcode == MMCO_SET_MAX_LONG &&

                                             long_arg == 16) &&

                         !(opcode == MMCO_LONG2UNUSED && FIELD_PICTURE(h)))) {

                        av_log(h->avctx, AV_LOG_ERROR,

                               "illegal long ref in memory management control "

                               "operation %d\n", opcode);

                        return -1;

                    }

                    mmco[i].long_arg = long_arg;

                }



                if (opcode > (unsigned) MMCO_LONG) {

                    av_log(h->avctx, AV_LOG_ERROR,

                           "illegal memory management control operation %d\n",

                           opcode);

                    return -1;

                }

                if (opcode == MMCO_END)

                    break;

            }

            mmco_index = i;

        } else {

            if (first_slice) {

                ret = ff_generate_sliding_window_mmcos(h, first_slice);

                if (ret < 0 && h->avctx->err_recognition & AV_EF_EXPLODE)

                    return ret;

            }

            mmco_index = -1;

        }

    }



    if (first_slice && mmco_index != -1) {

        memcpy(h->mmco, mmco_temp, sizeof(h->mmco));

        h->mmco_index = mmco_index;

    } else if (!first_slice && mmco_index >= 0 &&

               (mmco_index != h->mmco_index ||

                check_opcodes(h->mmco, mmco_temp, mmco_index))) {

        av_log(h->avctx, AV_LOG_ERROR,

               "Inconsistent MMCO state between slices [%d, %d]\n",

               mmco_index, h->mmco_index);

        return AVERROR_INVALIDDATA;

    }



    return 0;

}
