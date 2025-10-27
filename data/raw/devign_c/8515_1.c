SchroFrame *ff_create_schro_frame(AVCodecContext *avctx,

                                  SchroFrameFormat schro_frame_fmt)

{

    AVFrame *p_pic;

    SchroFrame *p_frame;

    int y_width, uv_width;

    int y_height, uv_height;

    int i;



    y_width   = avctx->width;

    y_height  = avctx->height;

    uv_width  = y_width  >> (SCHRO_FRAME_FORMAT_H_SHIFT(schro_frame_fmt));

    uv_height = y_height >> (SCHRO_FRAME_FORMAT_V_SHIFT(schro_frame_fmt));



    p_pic = av_frame_alloc();

    if (!p_pic)

        return NULL;



    if (ff_get_buffer(avctx, p_pic, AV_GET_BUFFER_FLAG_REF) < 0) {

        av_frame_free(&p_pic);

        return NULL;

    }



    p_frame         = schro_frame_new();

    p_frame->format = schro_frame_fmt;

    p_frame->width  = y_width;

    p_frame->height = y_height;

    schro_frame_set_free_callback(p_frame, free_schro_frame, p_pic);



    for (i = 0; i < 3; ++i) {

        p_frame->components[i].width  = i ? uv_width : y_width;

        p_frame->components[i].stride = p_pic->linesize[i];

        p_frame->components[i].height = i ? uv_height : y_height;

        p_frame->components[i].length =

                 p_frame->components[i].stride * p_frame->components[i].height;

        p_frame->components[i].data   = p_pic->data[i];



        if (i) {

            p_frame->components[i].v_shift =

                SCHRO_FRAME_FORMAT_V_SHIFT(p_frame->format);

            p_frame->components[i].h_shift =

                SCHRO_FRAME_FORMAT_H_SHIFT(p_frame->format);

        }

    }



    return p_frame;

}
