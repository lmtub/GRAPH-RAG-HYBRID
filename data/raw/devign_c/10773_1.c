static av_cold int svq1_encode_init(AVCodecContext *avctx)

{

    SVQ1Context * const s = avctx->priv_data;



    dsputil_init(&s->dsp, avctx);

    avctx->coded_frame= (AVFrame*)&s->picture;



    s->frame_width = avctx->width;

    s->frame_height = avctx->height;



    s->y_block_width = (s->frame_width + 15) / 16;

    s->y_block_height = (s->frame_height + 15) / 16;



    s->c_block_width = (s->frame_width / 4 + 15) / 16;

    s->c_block_height = (s->frame_height / 4 + 15) / 16;



    s->avctx= avctx;

    s->m.avctx= avctx;


    s->m.me.scratchpad= av_mallocz((avctx->width+64)*2*16*2*sizeof(uint8_t));

    s->m.me.map       = av_mallocz(ME_MAP_SIZE*sizeof(uint32_t));

    s->m.me.score_map = av_mallocz(ME_MAP_SIZE*sizeof(uint32_t));

    s->mb_type        = av_mallocz((s->y_block_width+1)*s->y_block_height*sizeof(int16_t));

    s->dummy          = av_mallocz((s->y_block_width+1)*s->y_block_height*sizeof(int32_t));

    h263_encode_init(&s->m); //mv_penalty



    return 0;

}