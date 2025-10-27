int ff_h264_queue_decode_slice(H264Context *h, const H2645NAL *nal)

{

    H264SliceContext *sl = h->slice_ctx + h->nb_slice_ctx_queued;

    int first_slice = sl == h->slice_ctx && !h->current_slice;

    int ret;



    sl->gb = nal->gb;



    ret = h264_slice_header_parse(h, sl, nal);

    if (ret < 0)

        return ret;



    // discard redundant pictures

    if (sl->redundant_pic_count > 0)

        return 0;



    if (sl->first_mb_addr == 0 || !h->current_slice) {

        if (h->setup_finished) {

            av_log(h->avctx, AV_LOG_ERROR, "Too many fields\n");

            return AVERROR_INVALIDDATA;

        }

    }



    if (sl->first_mb_addr == 0) { // FIXME better field boundary detection

        if (h->current_slice) {

            // this slice starts a new field

            // first decode any pending queued slices

            if (h->nb_slice_ctx_queued) {

                H264SliceContext tmp_ctx;



                ret = ff_h264_execute_decode_slices(h);

                if (ret < 0 && (h->avctx->err_recognition & AV_EF_EXPLODE))

                    return ret;



                memcpy(&tmp_ctx, h->slice_ctx, sizeof(tmp_ctx));

                memcpy(h->slice_ctx, sl, sizeof(tmp_ctx));

                memcpy(sl, &tmp_ctx, sizeof(tmp_ctx));

                sl = h->slice_ctx;

            }



            if (h->cur_pic_ptr && FIELD_PICTURE(h) && h->first_field) {

                ret = ff_h264_field_end(h, h->slice_ctx, 1);

                if (ret < 0)

                    return ret;

            } else if (h->cur_pic_ptr && !FIELD_PICTURE(h) && !h->first_field && h->nal_unit_type  == H264_NAL_IDR_SLICE) {

                av_log(h, AV_LOG_WARNING, "Broken frame packetizing\n");

                ret = ff_h264_field_end(h, h->slice_ctx, 1);

                ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX, 0);

                ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX, 1);

                h->cur_pic_ptr = NULL;

                if (ret < 0)

                    return ret;

            } else

                return AVERROR_INVALIDDATA;

        }



        if (!h->first_field) {

            if (h->cur_pic_ptr && !h->droppable) {

                ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX,

                                          h->picture_structure == PICT_BOTTOM_FIELD);

            }

            h->cur_pic_ptr = NULL;

        }

    }



    if (!h->current_slice)

        av_assert0(sl == h->slice_ctx);



    if (h->current_slice == 0 && !h->first_field) {

        if (

            (h->avctx->skip_frame >= AVDISCARD_NONREF && !h->nal_ref_idc) ||

            (h->avctx->skip_frame >= AVDISCARD_BIDIR  && sl->slice_type_nos == AV_PICTURE_TYPE_B) ||

            (h->avctx->skip_frame >= AVDISCARD_NONINTRA && sl->slice_type_nos != AV_PICTURE_TYPE_I) ||

            (h->avctx->skip_frame >= AVDISCARD_NONKEY && h->nal_unit_type != H264_NAL_IDR_SLICE && h->sei.recovery_point.recovery_frame_cnt < 0) ||

            h->avctx->skip_frame >= AVDISCARD_ALL) {

            return 0;

        }

    }



    if (!first_slice) {

        const PPS *pps = (const PPS*)h->ps.pps_list[sl->pps_id]->data;



        if (h->ps.pps->sps_id != pps->sps_id ||

            h->ps.pps->transform_8x8_mode != pps->transform_8x8_mode /*||

            (h->setup_finished && h->ps.pps != pps)*/) {

            av_log(h->avctx, AV_LOG_ERROR, "PPS changed between slices\n");

            return AVERROR_INVALIDDATA;

        }

        if (h->ps.sps != (const SPS*)h->ps.sps_list[h->ps.pps->sps_id]->data) {

            av_log(h->avctx, AV_LOG_ERROR,

               "SPS changed in the middle of the frame\n");

            return AVERROR_INVALIDDATA;

        }

    }



    if (h->current_slice == 0) {

        ret = h264_field_start(h, sl, nal, first_slice);

        if (ret < 0)

            return ret;

    } else {

        if (h->picture_structure != sl->picture_structure ||

            h->droppable         != (nal->ref_idc == 0)) {

            av_log(h->avctx, AV_LOG_ERROR,

                   "Changing field mode (%d -> %d) between slices is not allowed\n",

                   h->picture_structure, sl->picture_structure);

            return AVERROR_INVALIDDATA;

        } else if (!h->cur_pic_ptr) {

            av_log(h->avctx, AV_LOG_ERROR,

                   "unset cur_pic_ptr on slice %d\n",

                   h->current_slice + 1);

            return AVERROR_INVALIDDATA;

        }

    }



    ret = h264_slice_init(h, sl, nal);

    if (ret < 0)

        return ret;



    h->nb_slice_ctx_queued++;



    return 0;

}
