static void ipvideo_decode_format_10_opcodes(IpvideoContext *s, AVFrame *frame)

{

    int pass, x, y, changed_block;

    int16_t opcode, skip;

    GetByteContext decoding_map_ptr;

    GetByteContext skip_map_ptr;



    bytestream2_skip(&s->stream_ptr, 14); /* data starts 14 bytes in */



    /* this is PAL8, so make the palette available */

    memcpy(frame->data[1], s->pal, AVPALETTE_SIZE);

    s->stride = frame->linesize[0];



    s->line_inc = s->stride - 8;

    s->upper_motion_limit_offset = (s->avctx->height - 8) * frame->linesize[0]

                                  + (s->avctx->width - 8) * (1 + s->is_16bpp);



    bytestream2_init(&decoding_map_ptr, s->decoding_map, s->decoding_map_size);

    bytestream2_init(&skip_map_ptr, s->skip_map, s->skip_map_size);



    for (pass = 0; pass < 2; ++pass) {

        bytestream2_seek(&decoding_map_ptr, 0, SEEK_SET);

        bytestream2_seek(&skip_map_ptr, 0, SEEK_SET);

        skip = bytestream2_get_le16(&skip_map_ptr);



        for (y = 0; y < s->avctx->height; y += 8) {

            for (x = 0; x < s->avctx->width; x += 8) {

                s->pixel_ptr = s->cur_decode_frame->data[0] + x + y * s->cur_decode_frame->linesize[0];



                while (skip <= 0 && bytestream2_get_bytes_left(&decoding_map_ptr) > 1)  {

                    if (skip != -0x8000 && skip) {

                        opcode = bytestream2_get_le16(&decoding_map_ptr);

                        ipvideo_format_10_passes[pass](s, frame, opcode);

                        break;

                    }

                    skip = bytestream2_get_le16(&skip_map_ptr);

                }

                skip *= 2;

            }

        }

    }



    bytestream2_seek(&skip_map_ptr, 0, SEEK_SET);

    skip = bytestream2_get_le16(&skip_map_ptr);

    for (y = 0; y < s->avctx->height; y += 8) {

        for (x = 0; x < s->avctx->width; x += 8) {

            changed_block = 0;

            s->pixel_ptr = frame->data[0] + x + y*frame->linesize[0];



            while (skip <= 0)  {

                if (skip != -0x8000 && skip) {

                    changed_block = 1;

                    break;

                }

                if (bytestream2_get_bytes_left(&skip_map_ptr) < 2)

                    return;

                skip = bytestream2_get_le16(&skip_map_ptr);

            }



            if (changed_block) {

                copy_from(s, s->cur_decode_frame, frame, 0, 0);

            } else {

                /* Don't try to copy last_frame data on the first frame */

                if (s->avctx->frame_number)

                    copy_from(s, s->last_frame, frame, 0, 0);

            }

            skip *= 2;

        }

    }



    FFSWAP(AVFrame*, s->prev_decode_frame, s->cur_decode_frame);



    if (bytestream2_get_bytes_left(&s->stream_ptr) > 1) {

        av_log(s->avctx, AV_LOG_DEBUG,

               "decode finished with %d bytes left over\n",

               bytestream2_get_bytes_left(&s->stream_ptr));

    }

}
