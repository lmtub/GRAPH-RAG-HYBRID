int ff_h264_fill_default_ref_list(H264Context *h, H264SliceContext *sl)

{

    int i, len;

    int j;



    if (sl->slice_type_nos == AV_PICTURE_TYPE_B) {

        H264Picture *sorted[32];

        int cur_poc, list;

        int lens[2];



        if (FIELD_PICTURE(h))

            cur_poc = h->cur_pic_ptr->field_poc[h->picture_structure == PICT_BOTTOM_FIELD];

        else

            cur_poc = h->cur_pic_ptr->poc;



        for (list = 0; list < 2; list++) {

            len  = add_sorted(sorted,       h->short_ref, h->short_ref_count, cur_poc, 1 ^ list);

            len += add_sorted(sorted + len, h->short_ref, h->short_ref_count, cur_poc, 0 ^ list);

            av_assert0(len <= 32);



            len  = build_def_list(h->default_ref_list[list], FF_ARRAY_ELEMS(h->default_ref_list[0]),

                                  sorted, len, 0, h->picture_structure);

            len += build_def_list(h->default_ref_list[list] + len,

                                  FF_ARRAY_ELEMS(h->default_ref_list[0]) - len,

                                  h->long_ref, 16, 1, h->picture_structure);

            av_assert0(len <= 32);



            if (len < sl->ref_count[list])

                memset(&h->default_ref_list[list][len], 0, sizeof(H264Ref) * (sl->ref_count[list] - len));

            lens[list] = len;

        }



        if (lens[0] == lens[1] && lens[1] > 1) {

            for (i = 0; i < lens[0] &&

                        h->default_ref_list[0][i].parent->f->buf[0]->buffer ==

                        h->default_ref_list[1][i].parent->f->buf[0]->buffer; i++);

            if (i == lens[0]) {

                FFSWAP(H264Ref, h->default_ref_list[1][0], h->default_ref_list[1][1]);

            }

        }

    } else {

        len  = build_def_list(h->default_ref_list[0], FF_ARRAY_ELEMS(h->default_ref_list[0]),

                              h->short_ref, h->short_ref_count, 0, h->picture_structure);

        len += build_def_list(h->default_ref_list[0] + len,

                              FF_ARRAY_ELEMS(h->default_ref_list[0]) - len,

                              h-> long_ref, 16, 1, h->picture_structure);

        av_assert0(len <= 32);



        if (len < sl->ref_count[0])

            memset(&h->default_ref_list[0][len], 0, sizeof(H264Ref) * (sl->ref_count[0] - len));

    }

#ifdef TRACE

    for (i = 0; i < sl->ref_count[0]; i++) {

        ff_tlog(h->avctx, "List0: %s fn:%d 0x%p\n",

                h->default_ref_list[0][i].parent ? (h->default_ref_list[0][i].parent->long_ref ? "LT" : "ST") : "NULL",

                h->default_ref_list[0][i].pic_id,

                h->default_ref_list[0][i].parent ? h->default_ref_list[0][i].parent->f->data[0] : 0);

    }

    if (sl->slice_type_nos == AV_PICTURE_TYPE_B) {

        for (i = 0; i < sl->ref_count[1]; i++) {

            ff_tlog(h->avctx, "List1: %s fn:%d 0x%p\n",

                    h->default_ref_list[1][i].parent ? (h->default_ref_list[1][i].parent->long_ref ? "LT" : "ST") : "NULL",

                    h->default_ref_list[1][i].pic_id,

                    h->default_ref_list[1][i].parent ? h->default_ref_list[1][i].parent->f->data[0] : 0);

        }

    }

#endif



    for (j = 0; j<1+(sl->slice_type_nos == AV_PICTURE_TYPE_B); j++) {

        for (i = 0; i < sl->ref_count[j]; i++) {

            if (h->default_ref_list[j][i].parent) {

                AVFrame *f = h->default_ref_list[j][i].parent->f;

                if (h->cur_pic_ptr->f->width  != f->width ||

                    h->cur_pic_ptr->f->height != f->height ||

                    h->cur_pic_ptr->f->format != f->format) {

                    av_log(h->avctx, AV_LOG_ERROR, "Discarding mismatching reference\n");

                    memset(&h->default_ref_list[j][i], 0, sizeof(h->default_ref_list[j][i]));

                }

            }

        }

    }



    return 0;

}
