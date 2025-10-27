static int vp9_alloc_frame(AVCodecContext *ctx, VP9Frame *f)

{

    VP9Context *s = ctx->priv_data;

    int ret, sz;



    if ((ret = ff_thread_get_buffer(ctx, &f->tf, AV_GET_BUFFER_FLAG_REF)) < 0)

        return ret;

    sz = 64 * s->sb_cols * s->sb_rows;

    if (!(f->extradata = av_buffer_allocz(sz * (1 + sizeof(struct VP9mvrefPair))))) {

        ff_thread_release_buffer(ctx, &f->tf);

        return AVERROR(ENOMEM);

    }



    f->segmentation_map = f->extradata->data;

    f->mv = (struct VP9mvrefPair *) (f->extradata->data + sz);



    // retain segmentation map if it doesn't update

    if (s->segmentation.enabled && !s->segmentation.update_map &&

        !s->keyframe && !s->intraonly) {

        memcpy(f->segmentation_map, s->frames[LAST_FRAME].segmentation_map, sz);

    }



    return 0;

}
