static int alloc_frame_buffer(MpegEncContext *s, Picture *pic)

{

    int r;



    if (s->avctx->hwaccel) {

        assert(!pic->hwaccel_data_private);

        if (s->avctx->hwaccel->priv_data_size) {

            pic->hwaccel_data_private = av_malloc(s->avctx->hwaccel->priv_data_size);

            if (!pic->hwaccel_data_private) {

                av_log(s->avctx, AV_LOG_ERROR, "alloc_frame_buffer() failed (hwaccel private data allocation)\n");

                return -1;

            }

        }

    }



    r = s->avctx->get_buffer(s->avctx, (AVFrame*)pic);



    if (r<0 || !pic->age || !pic->type || !pic->data[0]) {

        av_log(s->avctx, AV_LOG_ERROR, "get_buffer() failed (%d %d %d %p)\n", r, pic->age, pic->type, pic->data[0]);

        av_freep(&pic->hwaccel_data_private);

        return -1;

    }



    if (s->linesize && (s->linesize != pic->linesize[0] || s->uvlinesize != pic->linesize[1])) {

        av_log(s->avctx, AV_LOG_ERROR, "get_buffer() failed (stride changed)\n");

        free_frame_buffer(s, pic);

        return -1;

    }



    if (pic->linesize[1] != pic->linesize[2]) {

        av_log(s->avctx, AV_LOG_ERROR, "get_buffer() failed (uv stride mismatch)\n");

        free_frame_buffer(s, pic);

        return -1;

    }



    return 0;

}
