static void vqa_decode_chunk(VqaContext *s)

{

    unsigned int chunk_type;

    unsigned int chunk_size;

    int byte_skip;

    unsigned int index = 0;

    int i;

    unsigned char r, g, b;

    int index_shift;



    int cbf0_chunk = -1;

    int cbfz_chunk = -1;

    int cbp0_chunk = -1;

    int cbpz_chunk = -1;

    int cpl0_chunk = -1;

    int cplz_chunk = -1;

    int vptz_chunk = -1;



    int x, y;

    int lines = 0;

    int pixel_ptr;

    int vector_index = 0;

    int lobyte = 0;

    int hibyte = 0;

    int lobytes = 0;

    int hibytes = s->decode_buffer_size / 2;



    /* first, traverse through the frame and find the subchunks */

    while (index < s->size) {



        chunk_type = AV_RB32(&s->buf[index]);

        chunk_size = AV_RB32(&s->buf[index + 4]);



        switch (chunk_type) {



        case CBF0_TAG:

            cbf0_chunk = index;

            break;



        case CBFZ_TAG:

            cbfz_chunk = index;

            break;



        case CBP0_TAG:

            cbp0_chunk = index;

            break;



        case CBPZ_TAG:

            cbpz_chunk = index;

            break;



        case CPL0_TAG:

            cpl0_chunk = index;

            break;



        case CPLZ_TAG:

            cplz_chunk = index;

            break;



        case VPTZ_TAG:

            vptz_chunk = index;

            break;



        default:

            av_log(s->avctx, AV_LOG_ERROR, "  VQA video: Found unknown chunk type: %c%c%c%c (%08X)\n",

            (chunk_type >> 24) & 0xFF,

            (chunk_type >> 16) & 0xFF,

            (chunk_type >>  8) & 0xFF,

            (chunk_type >>  0) & 0xFF,

            chunk_type);

            break;

        }



        byte_skip = chunk_size & 0x01;

        index += (CHUNK_PREAMBLE_SIZE + chunk_size + byte_skip);

    }



    /* next, deal with the palette */

    if ((cpl0_chunk != -1) && (cplz_chunk != -1)) {



        /* a chunk should not have both chunk types */

        av_log(s->avctx, AV_LOG_ERROR, "  VQA video: problem: found both CPL0 and CPLZ chunks\n");

        return;

    }



    /* decompress the palette chunk */

    if (cplz_chunk != -1) {



/* yet to be handled */



    }



    /* convert the RGB palette into the machine's endian format */

    if (cpl0_chunk != -1) {



        chunk_size = AV_RB32(&s->buf[cpl0_chunk + 4]);

        /* sanity check the palette size */

        if (chunk_size / 3 > 256) {

            av_log(s->avctx, AV_LOG_ERROR, "  VQA video: problem: found a palette chunk with %d colors\n",

                chunk_size / 3);

            return;

        }

        cpl0_chunk += CHUNK_PREAMBLE_SIZE;

        for (i = 0; i < chunk_size / 3; i++) {

            /* scale by 4 to transform 6-bit palette -> 8-bit */

            r = s->buf[cpl0_chunk++] * 4;

            g = s->buf[cpl0_chunk++] * 4;

            b = s->buf[cpl0_chunk++] * 4;

            s->palette[i] = (r << 16) | (g << 8) | (b);

        }

    }



    /* next, look for a full codebook */

    if ((cbf0_chunk != -1) && (cbfz_chunk != -1)) {



        /* a chunk should not have both chunk types */

        av_log(s->avctx, AV_LOG_ERROR, "  VQA video: problem: found both CBF0 and CBFZ chunks\n");

        return;

    }



    /* decompress the full codebook chunk */

    if (cbfz_chunk != -1) {



        chunk_size = AV_RB32(&s->buf[cbfz_chunk + 4]);

        cbfz_chunk += CHUNK_PREAMBLE_SIZE;

        decode_format80(&s->buf[cbfz_chunk], chunk_size,

            s->codebook, s->codebook_size, 0);

    }



    /* copy a full codebook */

    if (cbf0_chunk != -1) {



        chunk_size = AV_RB32(&s->buf[cbf0_chunk + 4]);

        /* sanity check the full codebook size */

        if (chunk_size > MAX_CODEBOOK_SIZE) {

            av_log(s->avctx, AV_LOG_ERROR, "  VQA video: problem: CBF0 chunk too large (0x%X bytes)\n",

                chunk_size);

            return;

        }

        cbf0_chunk += CHUNK_PREAMBLE_SIZE;



        memcpy(s->codebook, &s->buf[cbf0_chunk], chunk_size);

    }



    /* decode the frame */

    if (vptz_chunk == -1) {



        /* something is wrong if there is no VPTZ chunk */

        av_log(s->avctx, AV_LOG_ERROR, "  VQA video: problem: no VPTZ chunk found\n");

        return;

    }



    chunk_size = AV_RB32(&s->buf[vptz_chunk + 4]);

    vptz_chunk += CHUNK_PREAMBLE_SIZE;

    decode_format80(&s->buf[vptz_chunk], chunk_size,

        s->decode_buffer, s->decode_buffer_size, 1);



    /* render the final PAL8 frame */

    if (s->vector_height == 4)

        index_shift = 4;

    else

        index_shift = 3;

    for (y = 0; y < s->frame.linesize[0] * s->height;

        y += s->frame.linesize[0] * s->vector_height) {



        for (x = y; x < y + s->width; x += 4, lobytes++, hibytes++) {

            pixel_ptr = x;



            /* get the vector index, the method for which varies according to

             * VQA file version */

            switch (s->vqa_version) {



            case 1:

                lobyte = s->decode_buffer[lobytes * 2];

                hibyte = s->decode_buffer[(lobytes * 2) + 1];

                vector_index = ((hibyte << 8) | lobyte) >> 3;

                vector_index <<= index_shift;

                lines = s->vector_height;

                /* uniform color fill - a quick hack */

                if (hibyte == 0xFF) {

                    while (lines--) {

                        s->frame.data[0][pixel_ptr + 0] = 255 - lobyte;

                        s->frame.data[0][pixel_ptr + 1] = 255 - lobyte;

                        s->frame.data[0][pixel_ptr + 2] = 255 - lobyte;

                        s->frame.data[0][pixel_ptr + 3] = 255 - lobyte;

                        pixel_ptr += s->frame.linesize[0];

                    }

                    lines=0;

                }

                break;



            case 2:

                lobyte = s->decode_buffer[lobytes];

                hibyte = s->decode_buffer[hibytes];

                vector_index = (hibyte << 8) | lobyte;

                vector_index <<= index_shift;

                lines = s->vector_height;

                break;



            case 3:

/* not implemented yet */

                lines = 0;

                break;

            }



            while (lines--) {

                s->frame.data[0][pixel_ptr + 0] = s->codebook[vector_index++];

                s->frame.data[0][pixel_ptr + 1] = s->codebook[vector_index++];

                s->frame.data[0][pixel_ptr + 2] = s->codebook[vector_index++];

                s->frame.data[0][pixel_ptr + 3] = s->codebook[vector_index++];

                pixel_ptr += s->frame.linesize[0];

            }

        }

    }



    /* handle partial codebook */

    if ((cbp0_chunk != -1) && (cbpz_chunk != -1)) {

        /* a chunk should not have both chunk types */

        av_log(s->avctx, AV_LOG_ERROR, "  VQA video: problem: found both CBP0 and CBPZ chunks\n");

        return;

    }



    if (cbp0_chunk != -1) {



        chunk_size = AV_RB32(&s->buf[cbp0_chunk + 4]);

        cbp0_chunk += CHUNK_PREAMBLE_SIZE;



        /* accumulate partial codebook */

        memcpy(&s->next_codebook_buffer[s->next_codebook_buffer_index],

            &s->buf[cbp0_chunk], chunk_size);

        s->next_codebook_buffer_index += chunk_size;



        s->partial_countdown--;

        if (s->partial_countdown == 0) {



            /* time to replace codebook */

            memcpy(s->codebook, s->next_codebook_buffer,

                s->next_codebook_buffer_index);



            /* reset accounting */

            s->next_codebook_buffer_index = 0;

            s->partial_countdown = s->partial_count;

        }

    }



    if (cbpz_chunk != -1) {



        chunk_size = AV_RB32(&s->buf[cbpz_chunk + 4]);

        cbpz_chunk += CHUNK_PREAMBLE_SIZE;



        /* accumulate partial codebook */

        memcpy(&s->next_codebook_buffer[s->next_codebook_buffer_index],

            &s->buf[cbpz_chunk], chunk_size);

        s->next_codebook_buffer_index += chunk_size;



        s->partial_countdown--;

        if (s->partial_countdown == 0) {



            /* decompress codebook */

            decode_format80(s->next_codebook_buffer,

                s->next_codebook_buffer_index,

                s->codebook, s->codebook_size, 0);



            /* reset accounting */

            s->next_codebook_buffer_index = 0;

            s->partial_countdown = s->partial_count;

        }

    }

}
