int ff_xvmc_field_start(MpegEncContext*s, AVCodecContext *avctx)

{

    struct xvmc_pixfmt_render *last, *next, *render = (struct xvmc_pixfmt_render*)s->current_picture.data[2];

    const int mb_block_count = 4 + (1 << s->chroma_format);



    assert(avctx);

    if (!render || render->magic_id != AV_XVMC_RENDER_MAGIC ||

        !render->data_blocks || !render->mv_blocks){

        av_log(avctx, AV_LOG_ERROR,

               "Render token doesn't look as expected.\n");

        return -1; // make sure that this is a render packet

    }



    render->picture_structure = s->picture_structure;

    render->flags             = s->first_field ? 0 : XVMC_SECOND_FIELD;



    if (render->filled_mv_blocks_num) {

        av_log(avctx, AV_LOG_ERROR,

               "Rendering surface contains %i unprocessed blocks.\n",

               render->filled_mv_blocks_num);

        return -1;

    }

    if (render->total_number_of_mv_blocks   < 1 ||

        render->total_number_of_data_blocks < mb_block_count) {

        av_log(avctx, AV_LOG_ERROR,

               "Rendering surface doesn't provide enough block structures to work with.\n");

        return -1;

    }

    if (render->total_number_of_mv_blocks   < 1 ||

        render->total_number_of_data_blocks < mb_block_count) {

        av_log(avctx, AV_LOG_ERROR,

               "Rendering surface doesn't provide enough block structures to work with.\n");

        return -1;

    }



    render->p_future_surface = NULL;

    render->p_past_surface   = NULL;



    switch(s->pict_type) {

        case  FF_I_TYPE:

            return 0; // no prediction from other frames

        case  FF_B_TYPE:

            next = (struct xvmc_pixfmt_render*)s->next_picture.data[2];

            if (!next)

                return -1;

            if (next->magic_id != AV_XVMC_RENDER_MAGIC)

                return -1;

            render->p_future_surface = next->p_surface;

            // no return here, going to set forward prediction

        case  FF_P_TYPE:

            last = (struct xvmc_pixfmt_render*)s->last_picture.data[2];

            if (!last)

                last = render; // predict second field from the first

            if (last->magic_id != AV_XVMC_RENDER_MAGIC)

                return -1;

            render->p_past_surface = last->p_surface;

            return 0;

    }



return -1;

}
