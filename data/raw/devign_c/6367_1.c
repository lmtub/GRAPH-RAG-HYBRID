static int gif_image_write_image(ByteIOContext *pb, 

                                 int x1, int y1, int width, int height,

                                 const uint8_t *buf, int linesize, int pix_fmt)

{

    PutBitContext p;

    uint8_t buffer[200]; /* 100 * 9 / 8 = 113 */

    int i, left, w, v;

    const uint8_t *ptr;

    /* image block */



    put_byte(pb, 0x2c);

    put_le16(pb, x1);

    put_le16(pb, y1);

    put_le16(pb, width);

    put_le16(pb, height);

    put_byte(pb, 0x00); /* flags */

    /* no local clut */



    put_byte(pb, 0x08);



    left= width * height;



    init_put_bits(&p, buffer, 130);



/*

 * the thing here is the bitstream is written as little packets, with a size byte before

 * but it's still the same bitstream between packets (no flush !)

 */

    ptr = buf;

    w = width;

    while(left>0) {



        gif_put_bits_rev(&p, 9, 0x0100); /* clear code */



        for(i=0;i<GIF_CHUNKS;i++) {

            if (pix_fmt == PIX_FMT_RGB24) {

                v = gif_clut_index(ptr[0], ptr[1], ptr[2]);

                ptr+=3;

            } else {

                v = *ptr++;

            }

            gif_put_bits_rev(&p, 9, v);

            if (--w == 0) {

                w = width;

                buf += linesize;

                ptr = buf;

            }

        }



        if(left<=GIF_CHUNKS) {

            gif_put_bits_rev(&p, 9, 0x101); /* end of stream */

            gif_flush_put_bits_rev(&p);

        }

        if(pbBufPtr(&p) - p.buf > 0) {

            put_byte(pb, pbBufPtr(&p) - p.buf); /* byte count of the packet */

            put_buffer(pb, p.buf, pbBufPtr(&p) - p.buf); /* the actual buffer */

            p.buf_ptr = p.buf; /* dequeue the bytes off the bitstream */

        }

        if(left<=GIF_CHUNKS) {

            put_byte(pb, 0x00); /* end of image block */

        }



        left-=GIF_CHUNKS;

    }

    return 0;

}
