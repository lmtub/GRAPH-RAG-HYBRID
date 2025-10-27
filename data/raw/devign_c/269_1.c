int ff_h264_frame_start(H264Context *h)

{

    Picture *pic;

    int i, ret;

    const int pixel_shift = h->pixel_shift;

    int c[4] = {

        1<<(h->sps.bit_depth_luma-1),

        1<<(h->sps.bit_depth_chroma-1),

        1<<(h->sps.bit_depth_chroma-1),

        -1

    };



    if (!ff_thread_can_start_frame(h->avctx)) {

        av_log(h->avctx, AV_LOG_ERROR, "Attempt to start a frame outside SETUP state\n");

        return -1;

    }



    release_unused_pictures(h, 1);

    h->cur_pic_ptr = NULL;



    i = find_unused_picture(h);

    if (i < 0) {

        av_log(h->avctx, AV_LOG_ERROR, "no frame buffer available\n");

        return i;

    }

    pic = &h->DPB[i];



    pic->f.reference            = h->droppable ? 0 : h->picture_structure;

    pic->f.coded_picture_number = h->coded_picture_number++;

    pic->field_picture          = h->picture_structure != PICT_FRAME;



    /*

     * Zero key_frame here; IDR markings per slice in frame or fields are ORed

     * in later.

     * See decode_nal_units().

     */

    pic->f.key_frame = 0;

    pic->sync        = 0;

    pic->mmco_reset  = 0;



    if ((ret = alloc_picture(h, pic)) < 0)

        return ret;

    if(!h->sync && !h->avctx->hwaccel)

        avpriv_color_frame(&pic->f, c);



    h->cur_pic_ptr = pic;

    h->cur_pic     = *h->cur_pic_ptr;

    h->cur_pic.f.extended_data = h->cur_pic.f.data;



    ff_er_frame_start(&h->er);



    assert(h->linesize && h->uvlinesize);



    for (i = 0; i < 16; i++) {

        h->block_offset[i]           = (4 * ((scan8[i] - scan8[0]) & 7) << pixel_shift) + 4 * h->linesize * ((scan8[i] - scan8[0]) >> 3);

        h->block_offset[48 + i]      = (4 * ((scan8[i] - scan8[0]) & 7) << pixel_shift) + 8 * h->linesize * ((scan8[i] - scan8[0]) >> 3);

    }

    for (i = 0; i < 16; i++) {

        h->block_offset[16 + i]      =

        h->block_offset[32 + i]      = (4 * ((scan8[i] - scan8[0]) & 7) << pixel_shift) + 4 * h->uvlinesize * ((scan8[i] - scan8[0]) >> 3);

        h->block_offset[48 + 16 + i] =

        h->block_offset[48 + 32 + i] = (4 * ((scan8[i] - scan8[0]) & 7) << pixel_shift) + 8 * h->uvlinesize * ((scan8[i] - scan8[0]) >> 3);

    }



    /* can't be in alloc_tables because linesize isn't known there.

     * FIXME: redo bipred weight to not require extra buffer? */

    for (i = 0; i < h->slice_context_count; i++)

        if (h->thread_context[i]) {

            ret = alloc_scratch_buffers(h->thread_context[i], h->linesize);

            if (ret < 0)

                return ret;

        }



    /* Some macroblocks can be accessed before they're available in case

     * of lost slices, MBAFF or threading. */

    memset(h->slice_table, -1,

           (h->mb_height * h->mb_stride - 1) * sizeof(*h->slice_table));



    // s->decode = (h->flags & CODEC_FLAG_PSNR) || !s->encoding ||

    //             h->cur_pic.f.reference /* || h->contains_intra */ || 1;



    /* We mark the current picture as non-reference after allocating it, so

     * that if we break out due to an error it can be released automatically

     * in the next ff_MPV_frame_start().

     * SVQ3 as well as most other codecs have only last/next/current and thus

     * get released even with set reference, besides SVQ3 and others do not

     * mark frames as reference later "naturally". */

    if (h->avctx->codec_id != AV_CODEC_ID_SVQ3)

        h->cur_pic_ptr->f.reference = 0;



    h->cur_pic_ptr->field_poc[0] = h->cur_pic_ptr->field_poc[1] = INT_MAX;



    h->next_output_pic = NULL;



    assert(h->cur_pic_ptr->long_ref == 0);



    return 0;

}
