static int overlay_opencl_blend(FFFrameSync *fs)

{

    AVFilterContext    *avctx = fs->parent;

    AVFilterLink     *outlink = avctx->outputs[0];

    OverlayOpenCLContext *ctx = avctx->priv;

    AVFrame *input_main, *input_overlay;

    AVFrame *output;

    cl_mem mem;

    cl_int cle, x, y;

    size_t global_work[2];

    int kernel_arg = 0;

    int err, plane;



    err = ff_framesync_get_frame(fs, 0, &input_main, 0);

    if (err < 0)

        return err;

    err = ff_framesync_get_frame(fs, 1, &input_overlay, 0);

    if (err < 0)

        return err;



    if (!ctx->initialised) {

        AVHWFramesContext *main_fc =

            (AVHWFramesContext*)input_main->hw_frames_ctx->data;

        AVHWFramesContext *overlay_fc =

            (AVHWFramesContext*)input_overlay->hw_frames_ctx->data;



        err = overlay_opencl_load(avctx, main_fc->sw_format,

                                  overlay_fc->sw_format);

        if (err < 0)

            return err;

    }



    output = ff_get_video_buffer(outlink, outlink->w, outlink->h);

    if (!output) {

        err = AVERROR(ENOMEM);

        goto fail;

    }



    for (plane = 0; plane < ctx->nb_planes; plane++) {

        kernel_arg = 0;



        mem = (cl_mem)output->data[plane];

        cle = clSetKernelArg(ctx->kernel, kernel_arg++, sizeof(cl_mem), &mem);

        if (cle != CL_SUCCESS)

            goto fail_kernel_arg;



        mem = (cl_mem)input_main->data[plane];

        cle = clSetKernelArg(ctx->kernel, kernel_arg++, sizeof(cl_mem), &mem);

        if (cle != CL_SUCCESS)

            goto fail_kernel_arg;



        mem = (cl_mem)input_overlay->data[plane];

        cle = clSetKernelArg(ctx->kernel, kernel_arg++, sizeof(cl_mem), &mem);

        if (cle != CL_SUCCESS)

            goto fail_kernel_arg;



        if (ctx->alpha_separate) {

            mem = (cl_mem)input_overlay->data[ctx->nb_planes];

            cle = clSetKernelArg(ctx->kernel, kernel_arg++, sizeof(cl_mem), &mem);

            if (cle != CL_SUCCESS)

                goto fail_kernel_arg;

        }



        x = ctx->x_position / (plane == 0 ? 1 : ctx->x_subsample);

        y = ctx->y_position / (plane == 0 ? 1 : ctx->y_subsample);



        cle = clSetKernelArg(ctx->kernel, kernel_arg++, sizeof(cl_int), &x);

        if (cle != CL_SUCCESS)

            goto fail_kernel_arg;

        cle = clSetKernelArg(ctx->kernel, kernel_arg++, sizeof(cl_int), &y);

        if (cle != CL_SUCCESS)

            goto fail_kernel_arg;



        if (ctx->alpha_separate) {

            cl_int alpha_adj_x = plane == 0 ? 1 : ctx->x_subsample;

            cl_int alpha_adj_y = plane == 0 ? 1 : ctx->y_subsample;



            cle = clSetKernelArg(ctx->kernel, kernel_arg++, sizeof(cl_int), &alpha_adj_x);

            if (cle != CL_SUCCESS)

                goto fail_kernel_arg;

            cle = clSetKernelArg(ctx->kernel, kernel_arg++, sizeof(cl_int), &alpha_adj_y);

            if (cle != CL_SUCCESS)

                goto fail_kernel_arg;

        }



        global_work[0] = output->width;

        global_work[1] = output->height;



        cle = clEnqueueNDRangeKernel(ctx->command_queue, ctx->kernel, 2, NULL,

                                     global_work, NULL, 0, NULL, NULL);

        if (cle != CL_SUCCESS) {

            av_log(avctx, AV_LOG_ERROR, "Failed to enqueue "

                   "overlay kernel for plane %d: %d.\n", cle, plane);

            err = AVERROR(EIO);

            goto fail;

        }

    }



    cle = clFinish(ctx->command_queue);

    if (cle != CL_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR, "Failed to finish "

               "command queue: %d.\n", cle);

        err = AVERROR(EIO);

        goto fail;

    }



    err = av_frame_copy_props(output, input_main);



    av_log(avctx, AV_LOG_DEBUG, "Filter output: %s, %ux%u (%"PRId64").\n",

           av_get_pix_fmt_name(output->format),

           output->width, output->height, output->pts);



    return ff_filter_frame(outlink, output);



fail_kernel_arg:

    av_log(avctx, AV_LOG_ERROR, "Failed to set kernel arg %d: %d.\n",

           kernel_arg, cle);

    err = AVERROR(EIO);

fail:


    return err;

}