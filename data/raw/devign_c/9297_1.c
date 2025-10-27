static void ipvideo_decode_opcodes(IpvideoContext *s, AVFrame *frame)

{

    int x, y;

    unsigned char opcode;

    int ret;

    GetBitContext gb;



    bytestream2_skip(&s->stream_ptr, 14); /* data starts 14 bytes in */

    if (!s->is_16bpp) {

        /* this is PAL8, so make the palette available */

        memcpy(frame->data[1], s->pal, AVPALETTE_SIZE);



        s->stride = frame->linesize[0];

    } else {

        s->stride = frame->linesize[0] >> 1;

        s->mv_ptr = s->stream_ptr;

        bytestream2_skip(&s->mv_ptr, bytestream2_get_le16(&s->stream_ptr));

    }

    s->line_inc = s->stride - 8;

    s->upper_motion_limit_offset = (s->avctx->height - 8) * frame->linesize[0]

                                  + (s->avctx->width - 8) * (1 + s->is_16bpp);



    init_get_bits(&gb, s->decoding_map, s->decoding_map_size * 8);

    for (y = 0; y < s->avctx->height; y += 8) {

        for (x = 0; x < s->avctx->width; x += 8) {



            opcode = get_bits(&gb, 4);



            ff_tlog(s->avctx,

                    "  block @ (%3d, %3d): encoding 0x%X, data ptr offset %d\n",

                    x, y, opcode, bytestream2_tell(&s->stream_ptr));



            if (!s->is_16bpp) {

                s->pixel_ptr = frame->data[0] + x

                              + y*frame->linesize[0];

                ret = ipvideo_decode_block[opcode](s, frame);

            } else {

                s->pixel_ptr = frame->data[0] + x*2

                              + y*frame->linesize[0];

                ret = ipvideo_decode_block16[opcode](s, frame);

            }

            if (ret != 0) {

                av_log(s->avctx, AV_LOG_ERROR, "decode problem on frame %d, @ block (%d, %d)\n",

                       s->avctx->frame_number, x, y);


            }

        }

    }

    if (bytestream2_get_bytes_left(&s->stream_ptr) > 1) {

        av_log(s->avctx, AV_LOG_DEBUG,

               "decode finished with %d bytes left over\n",

               bytestream2_get_bytes_left(&s->stream_ptr));

    }

}