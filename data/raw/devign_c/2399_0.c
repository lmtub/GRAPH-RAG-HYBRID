static int scan_mmco_reset(AVCodecParserContext *s)

{

    H264ParseContext *p = s->priv_data;

    H264Context      *h = &p->h;

    H264SliceContext *sl = &h->slice_ctx[0];



    sl->slice_type_nos = s->pict_type & 3;



    if (h->pps.redundant_pic_cnt_present)

        get_ue_golomb(&sl->gb); // redundant_pic_count



    if (ff_set_ref_count(h, sl) < 0)

        return AVERROR_INVALIDDATA;



    if (sl->slice_type_nos != AV_PICTURE_TYPE_I) {

        int list;

        for (list = 0; list < sl->list_count; list++) {

            if (get_bits1(&sl->gb)) {

                int index;

                for (index = 0; ; index++) {

                    unsigned int reordering_of_pic_nums_idc = get_ue_golomb_31(&sl->gb);



                    if (reordering_of_pic_nums_idc < 3)

                        get_ue_golomb(&sl->gb);

                    else if (reordering_of_pic_nums_idc > 3) {

                        av_log(h->avctx, AV_LOG_ERROR,

                               "illegal reordering_of_pic_nums_idc %d\n",

                               reordering_of_pic_nums_idc);

                        return AVERROR_INVALIDDATA;

                    } else

                        break;



                    if (index >= sl->ref_count[list]) {

                        av_log(h->avctx, AV_LOG_ERROR,

                               "reference count %d overflow\n", index);

                        return AVERROR_INVALIDDATA;

                    }

                }

            }

        }

    }



    if ((h->pps.weighted_pred && sl->slice_type_nos == AV_PICTURE_TYPE_P) ||

        (h->pps.weighted_bipred_idc == 1 && sl->slice_type_nos == AV_PICTURE_TYPE_B))

        ff_pred_weight_table(h, sl);



    if (get_bits1(&sl->gb)) { // adaptive_ref_pic_marking_mode_flag

        int i;

        for (i = 0; i < MAX_MMCO_COUNT; i++) {

            MMCOOpcode opcode = get_ue_golomb_31(&sl->gb);

            if (opcode > (unsigned) MMCO_LONG) {

                av_log(h->avctx, AV_LOG_ERROR,

                       "illegal memory management control operation %d\n",

                       opcode);

                return AVERROR_INVALIDDATA;

            }

            if (opcode == MMCO_END)

               return 0;

            else if (opcode == MMCO_RESET)

                return 1;



            if (opcode == MMCO_SHORT2UNUSED || opcode == MMCO_SHORT2LONG)

                get_ue_golomb(&sl->gb);

            if (opcode == MMCO_SHORT2LONG || opcode == MMCO_LONG2UNUSED ||

                opcode == MMCO_LONG || opcode == MMCO_SET_MAX_LONG)

                get_ue_golomb_31(&sl->gb);

        }

    }



    return 0;

}
