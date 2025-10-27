int avcodec_copy_context(AVCodecContext *dest, const AVCodecContext *src)

{

    const AVCodec *orig_codec = dest->codec;

    uint8_t *orig_priv_data = dest->priv_data;



    if (avcodec_is_open(dest)) { // check that the dest context is uninitialized

        av_log(dest, AV_LOG_ERROR,

               "Tried to copy AVCodecContext %p into already-initialized %p\n",

               src, dest);

        return AVERROR(EINVAL);

    }



    copy_context_reset(dest);



    memcpy(dest, src, sizeof(*dest));

    av_opt_copy(dest, src);



    dest->priv_data       = orig_priv_data;

    dest->codec           = orig_codec;



    if (orig_priv_data && src->codec && src->codec->priv_class &&

        dest->codec && dest->codec->priv_class)

        av_opt_copy(orig_priv_data, src->priv_data);





    /* set values specific to opened codecs back to their default state */

    dest->slice_offset    = NULL;

    dest->hwaccel         = NULL;

    dest->internal        = NULL;

#if FF_API_CODED_FRAME

FF_DISABLE_DEPRECATION_WARNINGS

    dest->coded_frame     = NULL;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    /* reallocate values that should be allocated separately */

    dest->extradata       = NULL;


    dest->intra_matrix    = NULL;

    dest->inter_matrix    = NULL;

    dest->rc_override     = NULL;

    dest->subtitle_header = NULL;

    dest->hw_frames_ctx   = NULL;





#define alloc_and_copy_or_fail(obj, size, pad) \

    if (src->obj && size > 0) { \

        dest->obj = av_malloc(size + pad); \

        if (!dest->obj) \

            goto fail; \

        memcpy(dest->obj, src->obj, size); \

        if (pad) \

            memset(((uint8_t *) dest->obj) + size, 0, pad); \

    }

    alloc_and_copy_or_fail(extradata,    src->extradata_size,

                           AV_INPUT_BUFFER_PADDING_SIZE);

    dest->extradata_size  = src->extradata_size;

    alloc_and_copy_or_fail(intra_matrix, 64 * sizeof(int16_t), 0);

    alloc_and_copy_or_fail(inter_matrix, 64 * sizeof(int16_t), 0);

    alloc_and_copy_or_fail(rc_override,  src->rc_override_count * sizeof(*src->rc_override), 0);

    alloc_and_copy_or_fail(subtitle_header, src->subtitle_header_size, 1);

    av_assert0(dest->subtitle_header_size == src->subtitle_header_size);

#undef alloc_and_copy_or_fail



    if (src->hw_frames_ctx) {

        dest->hw_frames_ctx = av_buffer_ref(src->hw_frames_ctx);

        if (!dest->hw_frames_ctx)

            goto fail;

    }



    return 0;



fail:

    copy_context_reset(dest);

    return AVERROR(ENOMEM);

}