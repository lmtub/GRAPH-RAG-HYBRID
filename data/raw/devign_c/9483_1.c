int ff_mpeg_er_init(MpegEncContext *s)

{

    ERContext *er = &s->er;

    int mb_array_size = s->mb_height * s->mb_stride;

    int i;



    er->avctx       = s->avctx;



    er->mb_index2xy = s->mb_index2xy;

    er->mb_num      = s->mb_num;

    er->mb_width    = s->mb_width;

    er->mb_height   = s->mb_height;

    er->mb_stride   = s->mb_stride;

    er->b8_stride   = s->b8_stride;



    er->er_temp_buffer     = av_malloc(s->mb_height * s->mb_stride);

    er->error_status_table = av_mallocz(mb_array_size);

    if (!er->er_temp_buffer || !er->error_status_table)

        goto fail;



    er->mbskip_table  = s->mbskip_table;

    er->mbintra_table = s->mbintra_table;



    for (i = 0; i < FF_ARRAY_ELEMS(s->dc_val); i++)

        er->dc_val[i] = s->dc_val[i];



    er->decode_mb = mpeg_er_decode_mb;

    er->opaque    = s;



    return 0;

fail:

    av_freep(&er->er_temp_buffer);

    av_freep(&er->error_status_table);

    return AVERROR(ENOMEM);

}
