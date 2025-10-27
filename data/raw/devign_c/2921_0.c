static void decode_delta_e(uint8_t *dst,

                           const uint8_t *buf, const uint8_t *buf_end,

                           int w, int flag, int bpp, int dst_size)

{

    int planepitch = FFALIGN(w, 16) >> 3;

    int pitch = planepitch * bpp;

    int planepitch_byte = (w + 7) / 8;

    unsigned entries, ofssrc;

    GetByteContext gb, ptrs;

    PutByteContext pb;

    int k;



    if (buf_end - buf <= 4 * bpp)

        return;



    bytestream2_init_writer(&pb, dst, dst_size);

    bytestream2_init(&ptrs, buf, bpp * 4);



    for (k = 0; k < bpp; k++) {

        ofssrc = bytestream2_get_be32(&ptrs);



        if (!ofssrc)

            continue;



        if (ofssrc >= buf_end - buf)

            continue;



        bytestream2_init(&gb, buf + ofssrc, buf_end - (buf + ofssrc));



        entries = bytestream2_get_be16(&gb);

        while (entries) {

            int16_t opcode  = bytestream2_get_be16(&gb);

            unsigned offset = bytestream2_get_be32(&gb);



            bytestream2_seek_p(&pb, (offset / planepitch_byte) * pitch + (offset % planepitch_byte) + k * planepitch, SEEK_SET);

            if (opcode >= 0) {

                uint16_t x = bytestream2_get_be16(&gb);

                while (opcode && bytestream2_get_bytes_left_p(&pb) > 0) {

                    bytestream2_put_be16(&pb, x);

                    bytestream2_skip_p(&pb, pitch - 2);

                    opcode--;

                }

            } else {

                opcode = -opcode;

                while (opcode && bytestream2_get_bytes_left(&gb) > 0) {

                    bytestream2_put_be16(&pb, bytestream2_get_be16(&gb));

                    bytestream2_skip_p(&pb, pitch - 2);

                    opcode--;

                }

            }

            entries--;

        }

    }

}
