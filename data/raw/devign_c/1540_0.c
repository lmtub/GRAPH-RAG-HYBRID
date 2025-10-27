int ff_h264_decode_ref_pic_marking(const H264Context *h, H264SliceContext *sl,

                                   GetBitContext *gb)

{

    int i;

    MMCO *mmco = sl->mmco;

    int nb_mmco = 0;



    if (h->nal_unit_type == NAL_IDR_SLICE) { // FIXME fields

        skip_bits1(gb); // broken_link

        if (get_bits1(gb)) {

            mmco[0].opcode   = MMCO_LONG;

            mmco[0].long_arg = 0;

            nb_mmco          = 1;

        }

        sl->explicit_ref_marking = 1;

    } else {

        sl->explicit_ref_marking = get_bits1(gb);

        if (sl->explicit_ref_marking) {

            for (i = 0; i < MAX_MMCO_COUNT; i++) {

                MMCOOpcode opcode = get_ue_golomb_31(gb);



                mmco[i].opcode = opcode;

                if (opcode == MMCO_SHORT2UNUSED || opcode == MMCO_SHORT2LONG) {

                    mmco[i].short_pic_num =

                        (sl->curr_pic_num - get_ue_golomb(gb) - 1) &

                            (sl->max_pic_num - 1);

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

            nb_mmco = i;

        }

    }



    sl->nb_mmco = nb_mmco;



    return 0;

}
