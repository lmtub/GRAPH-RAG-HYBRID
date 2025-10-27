static int vaapi_encode_issue(AVCodecContext *avctx,

                              VAAPIEncodePicture *pic)

{

    VAAPIEncodeContext *ctx = avctx->priv_data;

    VAAPIEncodeSlice *slice;

    VAStatus vas;

    int err, i;

    char data[MAX_PARAM_BUFFER_SIZE];

    size_t bit_len;



    av_log(avctx, AV_LOG_DEBUG, "Issuing encode for pic %"PRId64"/%"PRId64" "

           "as type %s.\n", pic->display_order, pic->encode_order,

           picture_type_name[pic->type]);

    if (pic->nb_refs == 0) {

        av_log(avctx, AV_LOG_DEBUG, "No reference pictures.\n");

    } else {

        av_log(avctx, AV_LOG_DEBUG, "Refers to:");

        for (i = 0; i < pic->nb_refs; i++) {

            av_log(avctx, AV_LOG_DEBUG, " %"PRId64"/%"PRId64,

                   pic->refs[i]->display_order, pic->refs[i]->encode_order);

        }

        av_log(avctx, AV_LOG_DEBUG, ".\n");

    }



    av_assert0(pic->input_available && !pic->encode_issued);

    for (i = 0; i < pic->nb_refs; i++) {

        av_assert0(pic->refs[i]);

        // If we are serialised then the references must have already

        // completed.  If not, they must have been issued but need not

        // have completed yet.

        if (ctx->issue_mode == ISSUE_MODE_SERIALISE_EVERYTHING)

            av_assert0(pic->refs[i]->encode_complete);

        else

            av_assert0(pic->refs[i]->encode_issued);

    }



    av_log(avctx, AV_LOG_DEBUG, "Input surface is %#x.\n", pic->input_surface);



    pic->recon_image = av_frame_alloc();

    if (!pic->recon_image) {

        err = AVERROR(ENOMEM);

        goto fail;

    }



    err = av_hwframe_get_buffer(ctx->recon_frames_ref, pic->recon_image, 0);

    if (err < 0) {

        err = AVERROR(ENOMEM);

        goto fail;

    }

    pic->recon_surface = (VASurfaceID)(uintptr_t)pic->recon_image->data[3];

    av_log(avctx, AV_LOG_DEBUG, "Recon surface is %#x.\n", pic->recon_surface);



    pic->output_buffer_ref = av_buffer_pool_get(ctx->output_buffer_pool);

    if (!pic->output_buffer_ref) {

        err = AVERROR(ENOMEM);

        goto fail;

    }

    pic->output_buffer = (VABufferID)(uintptr_t)pic->output_buffer_ref->data;

    av_log(avctx, AV_LOG_DEBUG, "Output buffer is %#x.\n",

           pic->output_buffer);



    if (ctx->codec->picture_params_size > 0) {

        pic->codec_picture_params = av_malloc(ctx->codec->picture_params_size);

        if (!pic->codec_picture_params)

            goto fail;

        memcpy(pic->codec_picture_params, ctx->codec_picture_params,

               ctx->codec->picture_params_size);

    } else {

        av_assert0(!ctx->codec_picture_params);

    }



    pic->nb_param_buffers = 0;



    if (pic->encode_order == 0) {

        // Global parameter buffers are set on the first picture only.



        for (i = 0; i < ctx->nb_global_params; i++) {

            err = vaapi_encode_make_param_buffer(avctx, pic,

                                                 VAEncMiscParameterBufferType,

                                                 (char*)ctx->global_params[i],

                                                 ctx->global_params_size[i]);

            if (err < 0)

                goto fail;

        }

    }



    if (pic->type == PICTURE_TYPE_IDR && ctx->codec->init_sequence_params) {

        err = vaapi_encode_make_param_buffer(avctx, pic,

                                             VAEncSequenceParameterBufferType,

                                             ctx->codec_sequence_params,

                                             ctx->codec->sequence_params_size);

        if (err < 0)

            goto fail;

    }



    if (ctx->codec->init_picture_params) {

        err = ctx->codec->init_picture_params(avctx, pic);

        if (err < 0) {

            av_log(avctx, AV_LOG_ERROR, "Failed to initialise picture "

                   "parameters: %d.\n", err);

            goto fail;

        }

        err = vaapi_encode_make_param_buffer(avctx, pic,

                                             VAEncPictureParameterBufferType,

                                             pic->codec_picture_params,

                                             ctx->codec->picture_params_size);

        if (err < 0)

            goto fail;

    }



    if (pic->type == PICTURE_TYPE_IDR) {

        if (ctx->va_packed_headers & VA_ENC_PACKED_HEADER_SEQUENCE &&

            ctx->codec->write_sequence_header) {

            bit_len = 8 * sizeof(data);

            err = ctx->codec->write_sequence_header(avctx, data, &bit_len);

            if (err < 0) {

                av_log(avctx, AV_LOG_ERROR, "Failed to write per-sequence "

                       "header: %d.\n", err);

                goto fail;

            }

            err = vaapi_encode_make_packed_header(avctx, pic,

                                                  ctx->codec->sequence_header_type,

                                                  data, bit_len);

            if (err < 0)

                goto fail;

        }

    }



    if (ctx->va_packed_headers & VA_ENC_PACKED_HEADER_PICTURE &&

        ctx->codec->write_picture_header) {

        bit_len = 8 * sizeof(data);

        err = ctx->codec->write_picture_header(avctx, pic, data, &bit_len);

        if (err < 0) {

            av_log(avctx, AV_LOG_ERROR, "Failed to write per-picture "

                   "header: %d.\n", err);

            goto fail;

        }

        err = vaapi_encode_make_packed_header(avctx, pic,

                                              ctx->codec->picture_header_type,

                                              data, bit_len);

        if (err < 0)

            goto fail;

    }



    if (ctx->codec->write_extra_buffer) {

        for (i = 0;; i++) {

            size_t len = sizeof(data);

            int type;

            err = ctx->codec->write_extra_buffer(avctx, pic, i, &type,

                                                 data, &len);

            if (err == AVERROR_EOF)

                break;

            if (err < 0) {

                av_log(avctx, AV_LOG_ERROR, "Failed to write extra "

                       "buffer %d: %d.\n", i, err);

                goto fail;

            }



            err = vaapi_encode_make_param_buffer(avctx, pic, type,

                                                 data, len);

            if (err < 0)

                goto fail;

        }

    }



    if (ctx->va_packed_headers & VA_ENC_PACKED_HEADER_MISC &&

        ctx->codec->write_extra_header) {

        for (i = 0;; i++) {

            int type;

            bit_len = 8 * sizeof(data);

            err = ctx->codec->write_extra_header(avctx, pic, i, &type,

                                                 data, &bit_len);

            if (err == AVERROR_EOF)

                break;

            if (err < 0) {

                av_log(avctx, AV_LOG_ERROR, "Failed to write extra "

                       "header %d: %d.\n", i, err);

                goto fail;

            }



            err = vaapi_encode_make_packed_header(avctx, pic, type,

                                                  data, bit_len);

            if (err < 0)

                goto fail;

        }

    }



    av_assert0(pic->nb_slices <= MAX_PICTURE_SLICES);

    for (i = 0; i < pic->nb_slices; i++) {

        slice = av_mallocz(sizeof(*slice));

        if (!slice) {

            err = AVERROR(ENOMEM);

            goto fail;

        }

        slice->index = i;

        pic->slices[i] = slice;



        if (ctx->codec->slice_params_size > 0) {

            slice->codec_slice_params = av_mallocz(ctx->codec->slice_params_size);

            if (!slice->codec_slice_params) {

                err = AVERROR(ENOMEM);

                goto fail;

            }

        }



        if (ctx->codec->init_slice_params) {

            err = ctx->codec->init_slice_params(avctx, pic, slice);

            if (err < 0) {

                av_log(avctx, AV_LOG_ERROR, "Failed to initalise slice "

                       "parameters: %d.\n", err);

                goto fail;

            }

        }



        if (ctx->va_packed_headers & VA_ENC_PACKED_HEADER_SLICE &&

            ctx->codec->write_slice_header) {

            bit_len = 8 * sizeof(data);

            err = ctx->codec->write_slice_header(avctx, pic, slice,

                                                 data, &bit_len);

            if (err < 0) {

                av_log(avctx, AV_LOG_ERROR, "Failed to write per-slice "

                       "header: %d.\n", err);

                goto fail;

            }

            err = vaapi_encode_make_packed_header(avctx, pic,

                                                  ctx->codec->slice_header_type,

                                                  data, bit_len);

            if (err < 0)

                goto fail;

        }



        if (ctx->codec->init_slice_params) {

            err = vaapi_encode_make_param_buffer(avctx, pic,

                                                 VAEncSliceParameterBufferType,

                                                 slice->codec_slice_params,

                                                 ctx->codec->slice_params_size);

            if (err < 0)

                goto fail;

        }

    }



    vas = vaBeginPicture(ctx->hwctx->display, ctx->va_context,

                         pic->input_surface);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR, "Failed to begin picture encode issue: "

               "%d (%s).\n", vas, vaErrorStr(vas));

        err = AVERROR(EIO);

        goto fail_with_picture;

    }



    vas = vaRenderPicture(ctx->hwctx->display, ctx->va_context,

                          pic->param_buffers, pic->nb_param_buffers);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR, "Failed to upload encode parameters: "

               "%d (%s).\n", vas, vaErrorStr(vas));

        err = AVERROR(EIO);

        goto fail_with_picture;

    }



    vas = vaEndPicture(ctx->hwctx->display, ctx->va_context);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR, "Failed to end picture encode issue: "

               "%d (%s).\n", vas, vaErrorStr(vas));

        err = AVERROR(EIO);

        // vaRenderPicture() has been called here, so we should not destroy

        // the parameter buffers unless separate destruction is required.

        if (ctx->hwctx->driver_quirks &

            AV_VAAPI_DRIVER_QUIRK_RENDER_PARAM_BUFFERS)

            goto fail;

        else

            goto fail_at_end;

    }



    if (ctx->hwctx->driver_quirks &

        AV_VAAPI_DRIVER_QUIRK_RENDER_PARAM_BUFFERS) {

        for (i = 0; i < pic->nb_param_buffers; i++) {

            vas = vaDestroyBuffer(ctx->hwctx->display,

                                  pic->param_buffers[i]);

            if (vas != VA_STATUS_SUCCESS) {

                av_log(avctx, AV_LOG_ERROR, "Failed to destroy "

                       "param buffer %#x: %d (%s).\n",

                       pic->param_buffers[i], vas, vaErrorStr(vas));

                // And ignore.

            }

        }

    }



    pic->encode_issued = 1;



    if (ctx->issue_mode == ISSUE_MODE_SERIALISE_EVERYTHING)

        return vaapi_encode_wait(avctx, pic);

    else

        return 0;



fail_with_picture:

    vaEndPicture(ctx->hwctx->display, ctx->va_context);

fail:

    for(i = 0; i < pic->nb_param_buffers; i++)

        vaDestroyBuffer(ctx->hwctx->display, pic->param_buffers[i]);

fail_at_end:

    av_freep(&pic->codec_picture_params);

    av_frame_free(&pic->recon_image);



    return err;

}