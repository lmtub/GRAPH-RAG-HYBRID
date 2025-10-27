static int flac_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            uint8_t *buf, int buf_size)

{

    FLACContext *s = avctx->priv_data;

    int metadata_last, metadata_type, metadata_size;

    int tmp = 0, i, j = 0, input_buf_size;

    int16_t *samples = data, *left, *right;



    *data_size = 0;



    s->avctx = avctx;

    

    if(s->max_framesize == 0){

        s->max_framesize= 8192; // should hopefully be enough for the first header

        s->bitstream= av_fast_realloc(s->bitstream, &s->allocated_bitstream_size, s->max_framesize);

    }



    if(1 && s->max_framesize){//FIXME truncated

            buf_size= FFMIN(buf_size, s->max_framesize - s->bitstream_size);

            input_buf_size= buf_size;



            if(s->bitstream_index + s->bitstream_size + buf_size > s->allocated_bitstream_size){

//                printf("memmove\n");

                memmove(s->bitstream, &s->bitstream[s->bitstream_index], s->bitstream_size);

                s->bitstream_index=0;

            }

            memcpy(&s->bitstream[s->bitstream_index + s->bitstream_size], buf, buf_size);

            buf= &s->bitstream[s->bitstream_index];

            buf_size += s->bitstream_size;

            s->bitstream_size= buf_size;

            

            if(buf_size < s->max_framesize){

//                printf("wanna more data ...\n");

                return input_buf_size;

            }

    }



    init_get_bits(&s->gb, buf, buf_size*8);

    

    /* fLaC signature (be) */

    if (show_bits_long(&s->gb, 32) == bswap_32(ff_get_fourcc("fLaC")))

    {

        skip_bits(&s->gb, 32);



        av_log(s->avctx, AV_LOG_DEBUG, "STREAM HEADER\n");

        do {

            metadata_last = get_bits(&s->gb, 1);

            metadata_type = get_bits(&s->gb, 7);

            metadata_size = get_bits_long(&s->gb, 24);

            

            av_log(s->avctx, AV_LOG_DEBUG, " metadata block: flag = %d, type = %d, size = %d\n",

                metadata_last, metadata_type,

                metadata_size);

            if(metadata_size){

                switch(metadata_type)

                {

                case METADATA_TYPE_STREAMINFO:

                    if(metadata_size == 0)

                        av_log(s->avctx, AV_LOG_DEBUG, "size= 0 WTF!?\n");

                    

                    metadata_streaminfo(s);

                    dump_headers(s);

                    break;

                default:

                    for(i=0; i<metadata_size; i++)

                        skip_bits(&s->gb, 8);

                }

            }

        } while(!metadata_last);

    }

    else

    {

        

         tmp = show_bits(&s->gb, 16);

        if(tmp != 0xFFF8){

            av_log(s->avctx, AV_LOG_ERROR, "FRAME HEADER not here\n");

            while(get_bits_count(&s->gb)/8+2 < buf_size && show_bits(&s->gb, 16) != 0xFFF8)

                skip_bits(&s->gb, 8);

            goto end; // we may not have enough bits left to decode a frame, so try next time

        }

        skip_bits(&s->gb, 16);

        if (decode_frame(s) < 0)

            return -1;

    }



    

#if 0

    /* fix the channel order here */

    if (s->order == MID_SIDE)

    {

        short *left = samples;

        short *right = samples + s->blocksize;

        for (i = 0; i < s->blocksize; i += 2)

        {

            uint32_t x = s->decoded[0][i];

            uint32_t y = s->decoded[0][i+1];



            right[i] = x - (y / 2);

            left[i] = right[i] + y;

        }

        *data_size = 2 * s->blocksize;

    }

    else

    {

    for (i = 0; i < s->channels; i++)

    {

        switch(s->order)

        {

            case INDEPENDENT:

                for (j = 0; j < s->blocksize; j++)

                    samples[(s->blocksize*i)+j] = s->decoded[i][j];

                break;

            case LEFT_SIDE:

            case RIGHT_SIDE:

                if (i == 0)

                    for (j = 0; j < s->blocksize; j++)

                        samples[(s->blocksize*i)+j] = s->decoded[0][j];

                else

                    for (j = 0; j < s->blocksize; j++)

                        samples[(s->blocksize*i)+j] = s->decoded[0][j] - s->decoded[i][j];

                break;

//            case MID_SIDE:

//                av_log(s->avctx, AV_LOG_DEBUG, "mid-side unsupported\n");

        }

        *data_size += s->blocksize;

    }

    }

#else

    switch(s->order)

    {

        case INDEPENDENT:

            for (j = 0; j < s->blocksize; j++)

            {

                for (i = 0; i < s->channels; i++)

                    *(samples++) = s->decoded[i][j];

            }

            break;

        case LEFT_SIDE:

            assert(s->channels == 2);

            for (i = 0; i < s->blocksize; i++)

            {

                *(samples++) = s->decoded[0][i];

                *(samples++) = s->decoded[0][i] - s->decoded[1][i];

            }

            break;

        case RIGHT_SIDE:

            assert(s->channels == 2);

            for (i = 0; i < s->blocksize; i++)

            {

                *(samples++) = s->decoded[0][i] + s->decoded[1][i];

                *(samples++) = s->decoded[1][i];

            }

            break;

        case MID_SIDE:

            assert(s->channels == 2);

            for (i = 0; i < s->blocksize; i++)

            {

                int mid, side;

                mid = s->decoded[0][i];

                side = s->decoded[1][i];

                

                mid <<= 1;

                if (side & 1)

                    mid++;

                *(samples++) = (mid + side) >> 1;

                *(samples++) = (mid - side) >> 1;

            }

            break;

    }

#endif



    *data_size = (int8_t *)samples - (int8_t *)data;

    av_log(s->avctx, AV_LOG_DEBUG, "data size: %d\n", *data_size);



//    s->last_blocksize = s->blocksize;

end:

    i= (get_bits_count(&s->gb)+7)/8;;

    if(i > buf_size){

        av_log(s->avctx, AV_LOG_ERROR, "overread: %d\n", i - buf_size);

        return -1;

    }



    if(s->bitstream_size){

        s->bitstream_index += i;

        s->bitstream_size  -= i;

        return input_buf_size;

    }else 

        return i;

}
