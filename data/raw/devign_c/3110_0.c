static int decode_type1(GetByteContext *gb, PutByteContext *pb)

{

    unsigned opcode, len;

    int high = 0;

    int i, pos;



    while (bytestream2_get_bytes_left(gb) > 0) {

        GetByteContext gbc;



        while (bytestream2_get_bytes_left(gb) > 0) {

            while (bytestream2_get_bytes_left(gb) > 0) {

                opcode = bytestream2_get_byte(gb);

                high = opcode >= 0x20;

                if (high)

                    break;

                if (opcode)

                    break;

                opcode = bytestream2_get_byte(gb);

                if (opcode < 0xF8) {

                    opcode = opcode + 32;

                    break;

                }

                i = opcode - 0xF8;

                if (i) {

                    len = 256;

                    do {

                        len *= 2;

                        --i;

                    } while (i);

                } else {

                    len = 280;

                }

                do {

                    bytestream2_put_le32(pb, bytestream2_get_le32(gb));

                    bytestream2_put_le32(pb, bytestream2_get_le32(gb));

                    len -= 8;

                } while (len && bytestream2_get_bytes_left(gb) > 0);

            }



            if (!high) {

                do {

                    bytestream2_put_byte(pb, bytestream2_get_byte(gb));

                    --opcode;

                } while (opcode && bytestream2_get_bytes_left(gb) > 0);



                while (bytestream2_get_bytes_left(gb) > 0) {

                    GetByteContext gbc;



                    opcode = bytestream2_get_byte(gb);

                    if (opcode >= 0x20)

                        break;

                    bytestream2_init(&gbc, pb->buffer_start, pb->buffer_end - pb->buffer_start);



                    pos = -(opcode | 32 * bytestream2_get_byte(gb)) - 1;

                    bytestream2_seek(&gbc, bytestream2_tell_p(pb) + pos, SEEK_SET);

                    bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                    bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                    bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                    bytestream2_put_byte(pb, bytestream2_get_byte(gb));

                }

            }

            high = 0;

            if (opcode < 0x40)

                break;

            bytestream2_init(&gbc, pb->buffer_start, pb->buffer_end - pb->buffer_start);

            pos = (-((opcode & 0x1F) | 32 * bytestream2_get_byte(gb)) - 1);

            bytestream2_seek(&gbc, bytestream2_tell_p(pb) + pos, SEEK_SET);

            bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

            bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

            len = (opcode >> 5) - 1;

            do {

                bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                --len;

            } while (len && bytestream2_get_bytes_left(&gbc) > 0);

        }

        len = opcode & 0x1F;

        if (!len) {

            if (!bytestream2_peek_byte(gb)) {

                do {

                    bytestream2_skip(gb, 1);

                    len += 255;

                } while (!bytestream2_peek_byte(gb) && bytestream2_get_bytes_left(gb) > 0);

            }

            len += bytestream2_get_byte(gb) + 31;

        }

        pos = -bytestream2_get_byte(gb);

        bytestream2_init(&gbc, pb->buffer_start, pb->buffer_end - pb->buffer_start);

        bytestream2_seek(&gbc, bytestream2_tell_p(pb) + pos - (bytestream2_get_byte(gb) << 8), SEEK_SET);

        if (bytestream2_tell_p(pb) == bytestream2_tell(&gbc))

            break;

        if (len < 5 || bytestream2_tell_p(pb) - bytestream2_tell(&gbc) < 4) {

            bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

            bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

            bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

            do {

                bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                --len;

            } while (len && bytestream2_get_bytes_left(&gbc) > 0);

        } else {

            bytestream2_put_le32(pb, bytestream2_get_le32(&gbc));

            len--;

            do {

                bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                len--;

            } while (len && bytestream2_get_bytes_left(&gbc) > 0);

        }

    }



    return 0;

}
