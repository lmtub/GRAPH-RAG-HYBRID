static int read_ir(AVFilterLink *inlink, AVFrame *frame)

{

    AVFilterContext *ctx = inlink->dst;

    HeadphoneContext *s = ctx->priv;

    int ir_len, max_ir_len, input_number;



    for (input_number = 0; input_number < s->nb_inputs; input_number++)

        if (inlink == ctx->inputs[input_number])

            break;



    av_audio_fifo_write(s->in[input_number].fifo, (void **)frame->extended_data,

                        frame->nb_samples);

    av_frame_free(&frame);



    ir_len = av_audio_fifo_size(s->in[input_number].fifo);

    max_ir_len = 65536;

    if (ir_len > max_ir_len) {

        av_log(ctx, AV_LOG_ERROR, "Too big length of IRs: %d > %d.\n", ir_len, max_ir_len);

        return AVERROR(EINVAL);

    }

    s->in[input_number].ir_len = ir_len;

    s->ir_len = FFMAX(ir_len, s->ir_len);



    return 0;

}
