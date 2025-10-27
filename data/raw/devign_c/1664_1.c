static int alloc_picture(H264Context *h, Picture *pic)

{

    int i, ret = 0;



    av_assert0(!pic->f.data[0]);



    if (h->avctx->hwaccel) {

        const AVHWAccel *hwaccel = h->avctx->hwaccel;

        av_assert0(!pic->hwaccel_picture_private);

        if (hwaccel->priv_data_size) {

            pic->hwaccel_priv_buf = av_buffer_allocz(hwaccel->priv_data_size);

            if (!pic->hwaccel_priv_buf)

                return AVERROR(ENOMEM);

            pic->hwaccel_picture_private = pic->hwaccel_priv_buf->data;

        }

    }

    pic->tf.f = &pic->f;

    ret = ff_thread_get_buffer(h->avctx, &pic->tf, pic->reference ?

                                                   AV_GET_BUFFER_FLAG_REF : 0);

    if (ret < 0)

        goto fail;



    h->linesize   = pic->f.linesize[0];

    h->uvlinesize = pic->f.linesize[1];



    if (!h->qscale_table_pool) {

        ret = init_table_pools(h);

        if (ret < 0)

            goto fail;

    }



    pic->qscale_table_buf = av_buffer_pool_get(h->qscale_table_pool);

    pic->mb_type_buf      = av_buffer_pool_get(h->mb_type_pool);

    if (!pic->qscale_table_buf || !pic->mb_type_buf)

        goto fail;



    pic->mb_type      = (uint32_t*)pic->mb_type_buf->data + 2 * h->mb_stride + 1;

    pic->qscale_table = pic->qscale_table_buf->data + 2 * h->mb_stride + 1;



    for (i = 0; i < 2; i++) {

        pic->motion_val_buf[i] = av_buffer_pool_get(h->motion_val_pool);

        pic->ref_index_buf[i]  = av_buffer_pool_get(h->ref_index_pool);

        if (!pic->motion_val_buf[i] || !pic->ref_index_buf[i])

            goto fail;



        pic->motion_val[i] = (int16_t (*)[2])pic->motion_val_buf[i]->data + 4;

        pic->ref_index[i]  = pic->ref_index_buf[i]->data;

    }



    return 0;

fail:

    unref_picture(h, pic);

    return (ret < 0) ? ret : AVERROR(ENOMEM);

}
