static int decode_format80(VqaContext *s, int src_size,

    unsigned char *dest, int dest_size, int check_size) {



    int dest_index = 0;

    int count, opcode, start;

    int src_pos;

    unsigned char color;

    int i;



    start = bytestream2_tell(&s->gb);

    while (bytestream2_tell(&s->gb) - start < src_size) {

        opcode = bytestream2_get_byte(&s->gb);

        av_dlog(s->avctx, "opcode %02X: ", opcode);



        /* 0x80 means that frame is finished */

        if (opcode == 0x80)

            break;



        if (dest_index >= dest_size) {

            av_log(s->avctx, AV_LOG_ERROR, "decode_format80 problem: dest_index (%d) exceeded dest_size (%d)\n",

                dest_index, dest_size);

            return AVERROR_INVALIDDATA;

        }



        if (opcode == 0xFF) {



            count   = bytestream2_get_le16(&s->gb);

            src_pos = bytestream2_get_le16(&s->gb);

            av_dlog(s->avctx, "(1) copy %X bytes from absolute pos %X\n", count, src_pos);

            CHECK_COUNT();

            CHECK_COPY(src_pos);

            for (i = 0; i < count; i++)

                dest[dest_index + i] = dest[src_pos + i];

            dest_index += count;



        } else if (opcode == 0xFE) {



            count = bytestream2_get_le16(&s->gb);

            color = bytestream2_get_byte(&s->gb);

            av_dlog(s->avctx, "(2) set %X bytes to %02X\n", count, color);

            CHECK_COUNT();

            memset(&dest[dest_index], color, count);

            dest_index += count;



        } else if ((opcode & 0xC0) == 0xC0) {



            count = (opcode & 0x3F) + 3;

            src_pos = bytestream2_get_le16(&s->gb);

            av_dlog(s->avctx, "(3) copy %X bytes from absolute pos %X\n", count, src_pos);

            CHECK_COUNT();

            CHECK_COPY(src_pos);

            for (i = 0; i < count; i++)

                dest[dest_index + i] = dest[src_pos + i];

            dest_index += count;



        } else if (opcode > 0x80) {



            count = opcode & 0x3F;

            av_dlog(s->avctx, "(4) copy %X bytes from source to dest\n", count);

            CHECK_COUNT();

            bytestream2_get_buffer(&s->gb, &dest[dest_index], count);

            dest_index += count;



        } else {



            count = ((opcode & 0x70) >> 4) + 3;

            src_pos = bytestream2_get_byte(&s->gb) | ((opcode & 0x0F) << 8);

            av_dlog(s->avctx, "(5) copy %X bytes from relpos %X\n", count, src_pos);

            CHECK_COUNT();

            CHECK_COPY(dest_index - src_pos);

            for (i = 0; i < count; i++)

                dest[dest_index + i] = dest[dest_index - src_pos + i];

            dest_index += count;

        }

    }



    /* validate that the entire destination buffer was filled; this is

     * important for decoding frame maps since each vector needs to have a

     * codebook entry; it is not important for compressed codebooks because

     * not every entry needs to be filled */

    if (check_size)

        if (dest_index < dest_size)

            av_log(s->avctx, AV_LOG_ERROR, "decode_format80 problem: decode finished with dest_index (%d) < dest_size (%d)\n",

                dest_index, dest_size);



    return 0; // let's display what we decoded anyway

}
