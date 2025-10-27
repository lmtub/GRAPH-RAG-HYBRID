int ff_h264_decode_ref_pic_list_reordering(H264Context *h)

{

    int list, index, pic_structure, i;



    print_short_term(h);

    print_long_term(h);



    for (list = 0; list < h->list_count; list++) {

        for (i = 0; i < h->ref_count[list]; i++)

            COPY_PICTURE(&h->ref_list[list][i], &h->default_ref_list[list][i]);



        if (get_bits1(&h->gb)) {

            int pred = h->curr_pic_num;



            for (index = 0; ; index++) {

                unsigned int reordering_of_pic_nums_idc = get_ue_golomb_31(&h->gb);

                unsigned int pic_id;

                int i;

                Picture *ref = NULL;



                if (reordering_of_pic_nums_idc == 3)

                    break;



                if (index >= h->ref_count[list]) {

                    av_log(h->avctx, AV_LOG_ERROR, "reference count overflow\n");

                    return -1;

                }



                if (reordering_of_pic_nums_idc < 3) {

                    if (reordering_of_pic_nums_idc < 2) {

                        const unsigned int abs_diff_pic_num = get_ue_golomb(&h->gb) + 1;

                        int frame_num;



                        if (abs_diff_pic_num > h->max_pic_num) {

                            av_log(h->avctx, AV_LOG_ERROR, "abs_diff_pic_num overflow\n");

                            return -1;

                        }



                        if (reordering_of_pic_nums_idc == 0)

                            pred -= abs_diff_pic_num;

                        else

                            pred += abs_diff_pic_num;

                        pred &= h->max_pic_num - 1;



                        frame_num = pic_num_extract(h, pred, &pic_structure);



                        for (i = h->short_ref_count - 1; i >= 0; i--) {

                            ref = h->short_ref[i];

                            assert(ref->reference);

                            assert(!ref->long_ref);

                            if (ref->frame_num == frame_num &&

                                (ref->reference & pic_structure))

                                break;

                        }

                        if (i >= 0)

                            ref->pic_id = pred;

                    } else {

                        int long_idx;

                        pic_id = get_ue_golomb(&h->gb); //long_term_pic_idx



                        long_idx = pic_num_extract(h, pic_id, &pic_structure);



                        if (long_idx > 31) {

                            av_log(h->avctx, AV_LOG_ERROR, "long_term_pic_idx overflow\n");

                            return -1;

                        }

                        ref = h->long_ref[long_idx];

                        assert(!(ref && !ref->reference));

                        if (ref && (ref->reference & pic_structure)) {

                            ref->pic_id = pic_id;

                            assert(ref->long_ref);

                            i = 0;

                        } else {

                            i = -1;

                        }

                    }



                    if (i < 0) {

                        av_log(h->avctx, AV_LOG_ERROR, "reference picture missing during reorder\n");

                        memset(&h->ref_list[list][index], 0, sizeof(Picture)); //FIXME

                    } else {

                        for (i = index; i + 1 < h->ref_count[list]; i++) {

                            if (ref->long_ref == h->ref_list[list][i].long_ref &&

                                ref->pic_id   == h->ref_list[list][i].pic_id)

                                break;

                        }

                        for (; i > index; i--) {

                            COPY_PICTURE(&h->ref_list[list][i], &h->ref_list[list][i - 1]);

                        }

                        COPY_PICTURE(&h->ref_list[list][index], ref);

                        if (FIELD_PICTURE(h)) {

                            pic_as_field(&h->ref_list[list][index], pic_structure);

                        }

                    }

                } else {

                    av_log(h->avctx, AV_LOG_ERROR, "illegal reordering_of_pic_nums_idc\n");

                    return -1;

                }

            }

        }

    }

    for (list = 0; list < h->list_count; list++) {

        for (index = 0; index < h->ref_count[list]; index++) {

            if (!h->ref_list[list][index].f.data[0]) {

                av_log(h->avctx, AV_LOG_ERROR, "Missing reference picture\n");

                if (h->default_ref_list[list][0].f.data[0])

                    COPY_PICTURE(&h->ref_list[list][index], &h->default_ref_list[list][0]);

                else

                    return -1;

            }

        }

    }



    return 0;

}
