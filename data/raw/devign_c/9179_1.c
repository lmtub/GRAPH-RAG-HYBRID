int ff_xvmc_field_start(MpegEncContext *s, AVCodecContext *avctx)

{

    struct xvmc_pix_fmt *last, *next, *render = (struct xvmc_pix_fmt*)s->current_picture.f.data[2];

    const int mb_block_count = 4 + (1 << s->chroma_format);



    assert(avctx);

    if (!render || render->xvmc_id != AV_XVMC_ID ||

        !render->data_blocks || !render->mv_blocks ||

        (unsigned int)render->allocated_mv_blocks   > INT_MAX/(64*6) ||

        (unsigned int)render->allocated_data_blocks > INT_MAX/64     ||

        !render->p_surface) {

        av_log(avctx, AV_LOG_ERROR,

               "Render token doesn't look as expected.\n");

        return -1; // make sure that this is a render packet

    }



    if (render->filled_mv_blocks_num) {

        av_log(avctx, AV_LOG_ERROR,

               "Rendering surface contains %i unprocessed blocks.\n",

               render->filled_mv_blocks_num);

        return -1;

    }

    if (render->allocated_mv_blocks   < 1 ||

        render->allocated_data_blocks <  render->allocated_mv_blocks*mb_block_count ||

        render->start_mv_blocks_num   >= render->allocated_mv_blocks                ||

        render->next_free_data_block_num >

                        render->allocated_data_blocks -

                        mb_block_count*(render->allocated_mv_blocks-render->start_mv_blocks_num)) {

        av_log(avctx, AV_LOG_ERROR,

               "Rendering surface doesn't provide enough block structures to work with.\n");

        return -1;

    }



    render->picture_structure = s->picture_structure;

    render->flags             = s->first_field ? 0 : XVMC_SECOND_FIELD;

    render->p_future_surface  = NULL;

    render->p_past_surface    = NULL;



    switch(s->pict_type) {

        case  AV_PICTURE_TYPE_I:

            return 0; // no prediction from other frames

        case  AV_PICTURE_TYPE_B:

            next = (struct xvmc_pix_fmt*)s->next_picture.f.data[2];

            if (!next)

                return -1;

            if (next->xvmc_id != AV_XVMC_ID)

                return -1;

            render->p_future_surface = next->p_surface;

            // no return here, going to set forward prediction

        case  AV_PICTURE_TYPE_P:

            last = (struct xvmc_pix_fmt*)s->last_picture.f.data[2];

            if (!last)

                last = render; // predict second field from the first

            if (last->xvmc_id != AV_XVMC_ID)

                return -1;

            render->p_past_surface = last->p_surface;

            return 0;

    }



return -1;

}
