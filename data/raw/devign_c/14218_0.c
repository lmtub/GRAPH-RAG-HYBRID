static int update_frame_pool(AVCodecContext *avctx, AVFrame *frame)

{

    FramePool *pool = avctx->internal->pool;

    int i, ret;



    switch (avctx->codec_type) {

    case AVMEDIA_TYPE_VIDEO: {

        uint8_t *data[4];

        int linesize[4];

        int size[4] = { 0 };

        int w = frame->width;

        int h = frame->height;

        int tmpsize, unaligned;



        if (pool->format == frame->format &&

            pool->width == frame->width && pool->height == frame->height)

            return 0;



        avcodec_align_dimensions2(avctx, &w, &h, pool->stride_align);



        do {

            // NOTE: do not align linesizes individually, this breaks e.g. assumptions

            // that linesize[0] == 2*linesize[1] in the MPEG-encoder for 4:2:2

            av_image_fill_linesizes(linesize, avctx->pix_fmt, w);

            // increase alignment of w for next try (rhs gives the lowest bit set in w)

            w += w & ~(w - 1);



            unaligned = 0;

            for (i = 0; i < 4; i++)

                unaligned |= linesize[i] % pool->stride_align[i];

        } while (unaligned);



        tmpsize = av_image_fill_pointers(data, avctx->pix_fmt, h,

                                         NULL, linesize);

        if (tmpsize < 0)

            return -1;



        for (i = 0; i < 3 && data[i + 1]; i++)

            size[i] = data[i + 1] - data[i];

        size[i] = tmpsize - (data[i] - data[0]);



        for (i = 0; i < 4; i++) {

            av_buffer_pool_uninit(&pool->pools[i]);

            pool->linesize[i] = linesize[i];

            if (size[i]) {

                pool->pools[i] = av_buffer_pool_init(size[i] + 16 + STRIDE_ALIGN - 1,

                                                     CONFIG_MEMORY_POISONING ?

                                                        NULL :

                                                        av_buffer_allocz);

                if (!pool->pools[i]) {

                    ret = AVERROR(ENOMEM);

                    goto fail;

                }

            }

        }

        pool->format = frame->format;

        pool->width  = frame->width;

        pool->height = frame->height;



        break;

        }

    case AVMEDIA_TYPE_AUDIO: {

        int ch     = av_frame_get_channels(frame); //av_get_channel_layout_nb_channels(frame->channel_layout);

        int planar = av_sample_fmt_is_planar(frame->format);

        int planes = planar ? ch : 1;



        if (pool->format == frame->format && pool->planes == planes &&

            pool->channels == ch && frame->nb_samples == pool->samples)

            return 0;



        av_buffer_pool_uninit(&pool->pools[0]);

        ret = av_samples_get_buffer_size(&pool->linesize[0], ch,

                                         frame->nb_samples, frame->format, 0);

        if (ret < 0)

            goto fail;



        pool->pools[0] = av_buffer_pool_init(pool->linesize[0], NULL);

        if (!pool->pools[0]) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }



        pool->format     = frame->format;

        pool->planes     = planes;

        pool->channels   = ch;

        pool->samples = frame->nb_samples;

        break;

        }

    default: av_assert0(0);

    }

    return 0;

fail:

    for (i = 0; i < 4; i++)

        av_buffer_pool_uninit(&pool->pools[i]);

    pool->format = -1;

    pool->planes = pool->channels = pool->samples = 0;

    pool->width  = pool->height = 0;

    return ret;

}
