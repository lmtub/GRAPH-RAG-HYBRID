static int get_video_buffer(AVFrame *frame, int align)

{

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);

    int ret, i;



    if (!desc)

        return AVERROR(EINVAL);



    if ((ret = av_image_check_size(frame->width, frame->height, 0, NULL)) < 0)

        return ret;



    if (!frame->linesize[0]) {

        ret = av_image_fill_linesizes(frame->linesize, frame->format,

                                      frame->width);

        if (ret < 0)

            return ret;



        for (i = 0; i < 4 && frame->linesize[i]; i++)

            frame->linesize[i] = FFALIGN(frame->linesize[i], align);

    }



    for (i = 0; i < 4 && frame->linesize[i]; i++) {

        int h = frame->height;

        if (i == 1 || i == 2)

            h = -((-h) >> desc->log2_chroma_h);



        frame->buf[i] = av_buffer_alloc(frame->linesize[i] * h);

        if (!frame->buf[i])

            goto fail;



        frame->data[i] = frame->buf[i]->data;

    }

    if (desc->flags & PIX_FMT_PAL || desc->flags & PIX_FMT_PSEUDOPAL) {

        av_buffer_unref(&frame->buf[1]);

        frame->buf[1] = av_buffer_alloc(1024);

        if (!frame->buf[1])

            goto fail;

        frame->data[1] = frame->buf[1]->data;

    }



    frame->extended_data = frame->data;



    return 0;

fail:

    av_frame_unref(frame);

    return AVERROR(ENOMEM);

}
