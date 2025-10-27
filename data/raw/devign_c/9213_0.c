int ff_v4l2_context_set_status(V4L2Context* ctx, int cmd)

{

    int type = ctx->type;

    int ret;



    ret = ioctl(ctx_to_m2mctx(ctx)->fd, cmd, &type);

    if (ret < 0)

        return AVERROR(errno);



    ctx->streamon = (cmd == VIDIOC_STREAMON);



    return 0;

}
