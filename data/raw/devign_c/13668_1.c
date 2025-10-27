static int vc1_decode_sprites(VC1Context *v, GetBitContext* gb)

{

    int ret;

    MpegEncContext *s     = &v->s;

    AVCodecContext *avctx = s->avctx;

    SpriteData sd;



    memset(&sd, 0, sizeof(sd));



    ret = vc1_parse_sprites(v, gb, &sd);

    if (ret < 0)

        return ret;



    if (!s->current_picture.f->data[0]) {

        av_log(avctx, AV_LOG_ERROR, "Got no sprites\n");

        return -1;

    }



    if (v->two_sprites && (!s->last_picture_ptr || !s->last_picture.f->data[0])) {

        av_log(avctx, AV_LOG_WARNING, "Need two sprites, only got one\n");

        v->two_sprites = 0;

    }



    av_frame_unref(v->sprite_output_frame);

    if ((ret = ff_get_buffer(avctx, v->sprite_output_frame, 0)) < 0)

        return ret;



    vc1_draw_sprites(v, &sd);



    return 0;

}
