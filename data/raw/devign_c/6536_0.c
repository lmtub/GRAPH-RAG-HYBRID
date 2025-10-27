int ff_h264_update_thread_context(AVCodecContext *dst,

                                  const AVCodecContext *src)

{

    H264Context *h = dst->priv_data, *h1 = src->priv_data;

    int inited = h->context_initialized, err = 0;

    int need_reinit = 0;

    int i, ret;



    if (dst == src || !h1->context_initialized)

        return 0;



    if (!h1->ps.sps)

        return AVERROR_INVALIDDATA;



    if (inited &&

        (h->width                 != h1->width                 ||

         h->height                != h1->height                ||

         h->mb_width              != h1->mb_width              ||

         h->mb_height             != h1->mb_height             ||

         !h->ps.sps                                            ||

         h->ps.sps->bit_depth_luma    != h1->ps.sps->bit_depth_luma    ||

         h->ps.sps->chroma_format_idc != h1->ps.sps->chroma_format_idc ||

         h->ps.sps->colorspace        != h1->ps.sps->colorspace)) {

        need_reinit = 1;

    }



    // SPS/PPS

    for (i = 0; i < FF_ARRAY_ELEMS(h->ps.sps_list); i++) {

        av_buffer_unref(&h->ps.sps_list[i]);

        if (h1->ps.sps_list[i]) {

            h->ps.sps_list[i] = av_buffer_ref(h1->ps.sps_list[i]);

            if (!h->ps.sps_list[i])

                return AVERROR(ENOMEM);

        }

    }

    for (i = 0; i < FF_ARRAY_ELEMS(h->ps.pps_list); i++) {

        av_buffer_unref(&h->ps.pps_list[i]);

        if (h1->ps.pps_list[i]) {

            h->ps.pps_list[i] = av_buffer_ref(h1->ps.pps_list[i]);

            if (!h->ps.pps_list[i])

                return AVERROR(ENOMEM);

        }

    }



    h->ps.sps = h1->ps.sps;



    if (need_reinit || !inited) {

        h->width     = h1->width;

        h->height    = h1->height;

        h->mb_height = h1->mb_height;

        h->mb_width  = h1->mb_width;

        h->mb_num    = h1->mb_num;

        h->mb_stride = h1->mb_stride;

        h->b_stride  = h1->b_stride;



        if ((err = h264_slice_header_init(h)) < 0) {

            av_log(h->avctx, AV_LOG_ERROR, "h264_slice_header_init() failed");

            return err;

        }



        /* copy block_offset since frame_start may not be called */

        memcpy(h->block_offset, h1->block_offset, sizeof(h->block_offset));

    }



    h->avctx->coded_height  = h1->avctx->coded_height;

    h->avctx->coded_width   = h1->avctx->coded_width;

    h->avctx->width         = h1->avctx->width;

    h->avctx->height        = h1->avctx->height;

    h->coded_picture_number = h1->coded_picture_number;

    h->first_field          = h1->first_field;

    h->picture_structure    = h1->picture_structure;

    h->droppable            = h1->droppable;

    h->low_delay            = h1->low_delay;



    for (i = 0; i < H264_MAX_PICTURE_COUNT; i++) {

        ff_h264_unref_picture(h, &h->DPB[i]);

        if (h1->DPB[i].f->buf[0] &&

            (ret = ff_h264_ref_picture(h, &h->DPB[i], &h1->DPB[i])) < 0)

            return ret;

    }



    h->cur_pic_ptr = REBASE_PICTURE(h1->cur_pic_ptr, h, h1);

    ff_h264_unref_picture(h, &h->cur_pic);

    if (h1->cur_pic.f->buf[0]) {

        ret = ff_h264_ref_picture(h, &h->cur_pic, &h1->cur_pic);

        if (ret < 0)

            return ret;

    }



    h->enable_er       = h1->enable_er;

    h->workaround_bugs = h1->workaround_bugs;

    h->low_delay       = h1->low_delay;

    h->droppable       = h1->droppable;



    // extradata/NAL handling

    h->is_avc = h1->is_avc;

    h->nal_length_size = h1->nal_length_size;



    // POC timing

    copy_fields(h, h1, poc_lsb, current_slice);



    copy_picture_range(h->short_ref, h1->short_ref, 32, h, h1);

    copy_picture_range(h->long_ref, h1->long_ref, 32, h, h1);

    copy_picture_range(h->delayed_pic, h1->delayed_pic,

                       MAX_DELAYED_PIC_COUNT + 2, h, h1);



    if (!h->cur_pic_ptr)

        return 0;



    if (!h->droppable) {

        err = ff_h264_execute_ref_pic_marking(h, h->mmco, h->mmco_index);

        h->prev_poc_msb = h->poc_msb;

        h->prev_poc_lsb = h->poc_lsb;

    }

    h->prev_frame_num_offset = h->frame_num_offset;

    h->prev_frame_num        = h->frame_num;



    h->recovery_frame        = h1->recovery_frame;

    h->frame_recovered       = h1->frame_recovered;



    return err;

}
