static void xan_wc3_decode_frame(XanContext *s) {



    int width = s->avctx->width;

    int height = s->avctx->height;

    int total_pixels = width * height;

    unsigned char opcode;

    unsigned char flag = 0;

    int size = 0;

    int motion_x, motion_y;

    int x, y;



    unsigned char *opcode_buffer = s->buffer1;

    int opcode_buffer_size = s->buffer1_size;

    const unsigned char *imagedata_buffer = s->buffer2;



    /* pointers to segments inside the compressed chunk */

    const unsigned char *huffman_segment;

    const unsigned char *size_segment;

    const unsigned char *vector_segment;

    const unsigned char *imagedata_segment;



    huffman_segment =   s->buf + AV_RL16(&s->buf[0]);

    size_segment =      s->buf + AV_RL16(&s->buf[2]);

    vector_segment =    s->buf + AV_RL16(&s->buf[4]);

    imagedata_segment = s->buf + AV_RL16(&s->buf[6]);



    xan_huffman_decode(opcode_buffer, opcode_buffer_size,

                       huffman_segment, s->size - (huffman_segment - s->buf) );



    if (imagedata_segment[0] == 2)

        xan_unpack(s->buffer2, &imagedata_segment[1], s->buffer2_size);

    else

        imagedata_buffer = &imagedata_segment[1];



    /* use the decoded data segments to build the frame */

    x = y = 0;

    while (total_pixels) {



        opcode = *opcode_buffer++;

        size = 0;



        switch (opcode) {



        case 0:

            flag ^= 1;

            continue;



        case 1:

        case 2:

        case 3:

        case 4:

        case 5:

        case 6:

        case 7:

        case 8:

            size = opcode;

            break;



        case 12:

        case 13:

        case 14:

        case 15:

        case 16:

        case 17:

        case 18:

            size += (opcode - 10);

            break;



        case 9:

        case 19:

            size = *size_segment++;

            break;



        case 10:

        case 20:

            size = AV_RB16(&size_segment[0]);

            size_segment += 2;

            break;



        case 11:

        case 21:

            size = AV_RB24(size_segment);

            size_segment += 3;

            break;

        }



        if (opcode < 12) {

            flag ^= 1;

            if (flag) {

                /* run of (size) pixels is unchanged from last frame */

                xan_wc3_copy_pixel_run(s, x, y, size, 0, 0);

            } else {

                /* output a run of pixels from imagedata_buffer */

                xan_wc3_output_pixel_run(s, imagedata_buffer, x, y, size);

                imagedata_buffer += size;

            }

        } else {

            /* run-based motion compensation from last frame */

            motion_x = sign_extend(*vector_segment >> 4,  4);

            motion_y = sign_extend(*vector_segment & 0xF, 4);

            vector_segment++;



            /* copy a run of pixels from the previous frame */

            xan_wc3_copy_pixel_run(s, x, y, size, motion_x, motion_y);



            flag = 0;

        }



        /* coordinate accounting */

        total_pixels -= size;

        y += (x + size) / width;

        x  = (x + size) % width;

    }

}
