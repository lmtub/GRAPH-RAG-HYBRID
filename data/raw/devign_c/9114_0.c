static void mpeg_decode_sequence_extension(MpegEncContext *s)

{

    int horiz_size_ext, vert_size_ext;

    int bit_rate_ext;

    int level, profile;



    skip_bits(&s->gb, 1); /* profil and level esc*/

    profile= get_bits(&s->gb, 3);

    level= get_bits(&s->gb, 4);

    s->progressive_sequence = get_bits1(&s->gb); /* progressive_sequence */

    s->chroma_format = get_bits(&s->gb, 2); /* chroma_format 1=420, 2=422, 3=444 */

    horiz_size_ext = get_bits(&s->gb, 2);

    vert_size_ext = get_bits(&s->gb, 2);

    s->width |= (horiz_size_ext << 12);

    s->height |= (vert_size_ext << 12);

    bit_rate_ext = get_bits(&s->gb, 12);  /* XXX: handle it */

    s->bit_rate += (bit_rate_ext << 12) * 400;

    skip_bits1(&s->gb); /* marker */

    s->avctx->rc_buffer_size += get_bits(&s->gb, 8)*1024*16<<10;



    s->low_delay = get_bits1(&s->gb);

    if(s->flags & CODEC_FLAG_LOW_DELAY) s->low_delay=1;



    s->frame_rate_ext_n = get_bits(&s->gb, 2);

    s->frame_rate_ext_d = get_bits(&s->gb, 5);



    dprintf("sequence extension\n");

    s->codec_id= s->avctx->codec_id= CODEC_ID_MPEG2VIDEO;

    s->avctx->sub_id = 2; /* indicates mpeg2 found */



    if(s->avctx->debug & FF_DEBUG_PICT_INFO)

        av_log(s->avctx, AV_LOG_DEBUG, "profile: %d, level: %d vbv buffer: %d, bitrate:%d\n", 

               profile, level, s->avctx->rc_buffer_size, s->bit_rate);



}
