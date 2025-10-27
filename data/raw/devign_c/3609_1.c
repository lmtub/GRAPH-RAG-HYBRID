static int h264_field_start(H264Context *h, const H264SliceContext *sl,

                            const H2645NAL *nal, int first_slice)

{

    int i;

    const SPS *sps;



    int last_pic_structure, last_pic_droppable, ret;



    ret = h264_init_ps(h, sl, first_slice);

    if (ret < 0)

        return ret;



    sps = h->ps.sps;



    last_pic_droppable   = h->droppable;

    last_pic_structure   = h->picture_structure;

    h->droppable         = (nal->ref_idc == 0);

    h->picture_structure = sl->picture_structure;



    h->poc.frame_num        = sl->frame_num;

    h->poc.poc_lsb          = sl->poc_lsb;

    h->poc.delta_poc_bottom = sl->delta_poc_bottom;

    h->poc.delta_poc[0]     = sl->delta_poc[0];

    h->poc.delta_poc[1]     = sl->delta_poc[1];



    /* Shorten frame num gaps so we don't have to allocate reference

     * frames just to throw them away */

    if (h->poc.frame_num != h->poc.prev_frame_num) {

        int unwrap_prev_frame_num = h->poc.prev_frame_num;

        int max_frame_num         = 1 << sps->log2_max_frame_num;



        if (unwrap_prev_frame_num > h->poc.frame_num)

            unwrap_prev_frame_num -= max_frame_num;



        if ((h->poc.frame_num - unwrap_prev_frame_num) > sps->ref_frame_count) {

            unwrap_prev_frame_num = (h->poc.frame_num - sps->ref_frame_count) - 1;

            if (unwrap_prev_frame_num < 0)

                unwrap_prev_frame_num += max_frame_num;



            h->poc.prev_frame_num = unwrap_prev_frame_num;

        }

    }



    /* See if we have a decoded first field looking for a pair...

     * Here, we're using that to see if we should mark previously

     * decode frames as "finished".

     * We have to do that before the "dummy" in-between frame allocation,

     * since that can modify h->cur_pic_ptr. */

    if (h->first_field) {

        av_assert0(h->cur_pic_ptr);

        av_assert0(h->cur_pic_ptr->f->buf[0]);

        assert(h->cur_pic_ptr->reference != DELAYED_PIC_REF);



        /* Mark old field/frame as completed */

        if (h->cur_pic_ptr->tf.owner == h->avctx) {

            ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX,

                                      last_pic_structure == PICT_BOTTOM_FIELD);

        }



        /* figure out if we have a complementary field pair */

        if (!FIELD_PICTURE(h) || h->picture_structure == last_pic_structure) {

            /* Previous field is unmatched. Don't display it, but let it

             * remain for reference if marked as such. */

            if (last_pic_structure != PICT_FRAME) {

                ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX,

                                          last_pic_structure == PICT_TOP_FIELD);

            }

        } else {

            if (h->cur_pic_ptr->frame_num != h->poc.frame_num) {

                /* This and previous field were reference, but had

                 * different frame_nums. Consider this field first in

                 * pair. Throw away previous field except for reference

                 * purposes. */

                if (last_pic_structure != PICT_FRAME) {

                    ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX,

                                              last_pic_structure == PICT_TOP_FIELD);

                }

            } else {

                /* Second field in complementary pair */

                if (!((last_pic_structure   == PICT_TOP_FIELD &&

                       h->picture_structure == PICT_BOTTOM_FIELD) ||

                      (last_pic_structure   == PICT_BOTTOM_FIELD &&

                       h->picture_structure == PICT_TOP_FIELD))) {

                    av_log(h->avctx, AV_LOG_ERROR,

                           "Invalid field mode combination %d/%d\n",

                           last_pic_structure, h->picture_structure);

                    h->picture_structure = last_pic_structure;

                    h->droppable         = last_pic_droppable;

                    return AVERROR_INVALIDDATA;

                } else if (last_pic_droppable != h->droppable) {

                    avpriv_request_sample(h->avctx,

                                          "Found reference and non-reference fields in the same frame, which");

                    h->picture_structure = last_pic_structure;

                    h->droppable         = last_pic_droppable;

                    return AVERROR_PATCHWELCOME;

                }

            }

        }

    }



    while (h->poc.frame_num != h->poc.prev_frame_num && !h->first_field &&

           h->poc.frame_num != (h->poc.prev_frame_num + 1) % (1 << sps->log2_max_frame_num)) {

        H264Picture *prev = h->short_ref_count ? h->short_ref[0] : NULL;

        av_log(h->avctx, AV_LOG_DEBUG, "Frame num gap %d %d\n",

               h->poc.frame_num, h->poc.prev_frame_num);

        if (!sps->gaps_in_frame_num_allowed_flag)

            for(i=0; i<FF_ARRAY_ELEMS(h->last_pocs); i++)

                h->last_pocs[i] = INT_MIN;

        ret = h264_frame_start(h);

        if (ret < 0) {

            h->first_field = 0;

            return ret;

        }



        h->poc.prev_frame_num++;

        h->poc.prev_frame_num        %= 1 << sps->log2_max_frame_num;

        h->cur_pic_ptr->frame_num = h->poc.prev_frame_num;

        h->cur_pic_ptr->invalid_gap = !sps->gaps_in_frame_num_allowed_flag;

        ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX, 0);

        ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX, 1);



        h->explicit_ref_marking = 0;

        ret = ff_h264_execute_ref_pic_marking(h);

        if (ret < 0 && (h->avctx->err_recognition & AV_EF_EXPLODE))

            return ret;

        /* Error concealment: If a ref is missing, copy the previous ref

         * in its place.

         * FIXME: Avoiding a memcpy would be nice, but ref handling makes

         * many assumptions about there being no actual duplicates.

         * FIXME: This does not copy padding for out-of-frame motion

         * vectors.  Given we are concealing a lost frame, this probably

         * is not noticeable by comparison, but it should be fixed. */

        if (h->short_ref_count) {

            if (prev &&

                h->short_ref[0]->f->width == prev->f->width &&

                h->short_ref[0]->f->height == prev->f->height &&

                h->short_ref[0]->f->format == prev->f->format) {

                ff_thread_await_progress(&prev->tf, INT_MAX, 0);

                if (prev->field_picture)

                    ff_thread_await_progress(&prev->tf, INT_MAX, 1);

                av_image_copy(h->short_ref[0]->f->data,

                              h->short_ref[0]->f->linesize,

                              (const uint8_t **)prev->f->data,

                              prev->f->linesize,

                              prev->f->format,

                              prev->f->width,

                              prev->f->height);

                h->short_ref[0]->poc = prev->poc + 2;

            }

            h->short_ref[0]->frame_num = h->poc.prev_frame_num;

        }

    }



    /* See if we have a decoded first field looking for a pair...

     * We're using that to see whether to continue decoding in that

     * frame, or to allocate a new one. */

    if (h->first_field) {

        av_assert0(h->cur_pic_ptr);

        av_assert0(h->cur_pic_ptr->f->buf[0]);

        assert(h->cur_pic_ptr->reference != DELAYED_PIC_REF);



        /* figure out if we have a complementary field pair */

        if (!FIELD_PICTURE(h) || h->picture_structure == last_pic_structure) {

            /* Previous field is unmatched. Don't display it, but let it

             * remain for reference if marked as such. */

            h->missing_fields ++;

            h->cur_pic_ptr = NULL;

            h->first_field = FIELD_PICTURE(h);

        } else {

            h->missing_fields = 0;

            if (h->cur_pic_ptr->frame_num != h->poc.frame_num) {

                ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX,

                                          h->picture_structure==PICT_BOTTOM_FIELD);

                /* This and the previous field had different frame_nums.

                 * Consider this field first in pair. Throw away previous

                 * one except for reference purposes. */

                h->first_field = 1;

                h->cur_pic_ptr = NULL;

            } else {

                /* Second field in complementary pair */

                h->first_field = 0;

            }

        }

    } else {

        /* Frame or first field in a potentially complementary pair */

        h->first_field = FIELD_PICTURE(h);

    }



    if (!FIELD_PICTURE(h) || h->first_field) {

        if (h264_frame_start(h) < 0) {

            h->first_field = 0;

            return AVERROR_INVALIDDATA;

        }

    } else {

        release_unused_pictures(h, 0);

    }

    /* Some macroblocks can be accessed before they're available in case

    * of lost slices, MBAFF or threading. */

    if (FIELD_PICTURE(h)) {

        for(i = (h->picture_structure == PICT_BOTTOM_FIELD); i<h->mb_height; i++)

            memset(h->slice_table + i*h->mb_stride, -1, (h->mb_stride - (i+1==h->mb_height)) * sizeof(*h->slice_table));

    } else {

        memset(h->slice_table, -1,

            (h->mb_height * h->mb_stride - 1) * sizeof(*h->slice_table));

    }



    ff_h264_init_poc(h->cur_pic_ptr->field_poc, &h->cur_pic_ptr->poc,

                     h->ps.sps, &h->poc, h->picture_structure, nal->ref_idc);



    memcpy(h->mmco, sl->mmco, sl->nb_mmco * sizeof(*h->mmco));

    h->nb_mmco = sl->nb_mmco;

    h->explicit_ref_marking = sl->explicit_ref_marking;



    h->picture_idr = nal->type == H264_NAL_IDR_SLICE;



    if (h->sei.recovery_point.recovery_frame_cnt >= 0) {

        const int sei_recovery_frame_cnt = h->sei.recovery_point.recovery_frame_cnt;



        if (h->poc.frame_num != sei_recovery_frame_cnt || sl->slice_type_nos != AV_PICTURE_TYPE_I)

            h->valid_recovery_point = 1;



        if (   h->recovery_frame < 0

            || av_mod_uintp2(h->recovery_frame - h->poc.frame_num, h->ps.sps->log2_max_frame_num) > sei_recovery_frame_cnt) {

            h->recovery_frame = av_mod_uintp2(h->poc.frame_num + sei_recovery_frame_cnt, h->ps.sps->log2_max_frame_num);



            if (!h->valid_recovery_point)

                h->recovery_frame = h->poc.frame_num;

        }

    }



    h->cur_pic_ptr->f->key_frame |= (nal->type == H264_NAL_IDR_SLICE);



    if (nal->type == H264_NAL_IDR_SLICE ||

        (h->recovery_frame == h->poc.frame_num && nal->ref_idc)) {

        h->recovery_frame         = -1;

        h->cur_pic_ptr->recovered = 1;

    }

    // If we have an IDR, all frames after it in decoded order are

    // "recovered".

    if (nal->type == H264_NAL_IDR_SLICE)

        h->frame_recovered |= FRAME_RECOVERED_IDR;

#if 1

    h->cur_pic_ptr->recovered |= h->frame_recovered;

#else

    h->cur_pic_ptr->recovered |= !!(h->frame_recovered & FRAME_RECOVERED_IDR);

#endif



    /* Set the frame properties/side data. Only done for the second field in

     * field coded frames, since some SEI information is present for each field

     * and is merged by the SEI parsing code. */

    if (!FIELD_PICTURE(h) || !h->first_field || h->missing_fields > 1) {

        ret = h264_export_frame_props(h);

        if (ret < 0)

            return ret;



        ret = h264_select_output_frame(h);

        if (ret < 0)

            return ret;

    }



    return 0;

}
