static int encode_mode(CinepakEncContext *s, CinepakMode mode, int h, int v1_size, int v4_size, int v4, AVPicture *scratch_pict, strip_info *info, unsigned char *buf)

{

    int x, y, z, flags, bits, temp_size, header_ofs, ret = 0, mb_count = s->w * h / MB_AREA;

    int needs_extra_bit, should_write_temp;

    unsigned char temp[64]; //32/2 = 16 V4 blocks at 4 B each -> 64 B

    mb_info *mb;

    AVPicture sub_scratch;



    //encode codebooks

    if(v1_size)

        ret += encode_codebook(s, info->v1_codebook, v1_size, 0x22, 0x26, buf + ret);



    if(v4_size)

        ret += encode_codebook(s, info->v4_codebook, v4_size, 0x20, 0x24, buf + ret);



    //update scratch picture

    for(z = y = 0; y < h; y += MB_SIZE) {

        for(x = 0; x < s->w; x += MB_SIZE, z++) {

            mb = &s->mb[z];



            if(mode == MODE_MC && mb->best_encoding == ENC_SKIP)

                continue;



            get_sub_picture(s, x, y, scratch_pict, &sub_scratch);



            if(mode == MODE_V1_ONLY || mb->best_encoding == ENC_V1)

                decode_v1_vector(s, &sub_scratch, mb, info);

            else if(mode != MODE_V1_ONLY && mb->best_encoding == ENC_V4)

                decode_v4_vector(s, &sub_scratch, mb->v4_vector[v4], info);

        }

    }



    switch(mode) {

    case MODE_V1_ONLY:

        //av_log(s->avctx, AV_LOG_INFO, "mb_count = %i\n", mb_count);

        ret += write_chunk_header(buf + ret, 0x32, mb_count);



        for(x = 0; x < mb_count; x++)

            buf[ret++] = s->mb[x].v1_vector;



        break;

    case MODE_V1_V4:

        //remember header position

        header_ofs = ret;

        ret += CHUNK_HEADER_SIZE;



        for(x = 0; x < mb_count; x += 32) {

            flags = 0;

            for(y = x; y < FFMIN(x+32, mb_count); y++)

                if(s->mb[y].best_encoding == ENC_V4)

                    flags |= 1 << (31 - y + x);



            AV_WB32(&buf[ret], flags);

            ret += 4;



            for(y = x; y < FFMIN(x+32, mb_count); y++) {

                mb = &s->mb[y];



                if(mb->best_encoding == ENC_V1)

                    buf[ret++] = mb->v1_vector;

                else

                    for(z = 0; z < 4; z++)

                        buf[ret++] = mb->v4_vector[v4][z];

            }

        }



        write_chunk_header(buf + header_ofs, 0x30, ret - header_ofs - CHUNK_HEADER_SIZE);



        break;

    case MODE_MC:

        //remember header position

        header_ofs = ret;

        ret += CHUNK_HEADER_SIZE;

        flags = bits = temp_size = 0;



        for(x = 0; x < mb_count; x++) {

            mb = &s->mb[x];

            flags |= (mb->best_encoding != ENC_SKIP) << (31 - bits++);

            needs_extra_bit = 0;

            should_write_temp = 0;



            if(mb->best_encoding != ENC_SKIP) {

                if(bits < 32)

                    flags |= (mb->best_encoding == ENC_V4) << (31 - bits++);

                else

                    needs_extra_bit = 1;

            }



            if(bits == 32) {

                AV_WB32(&buf[ret], flags);

                ret += 4;

                flags = bits = 0;



                if(mb->best_encoding == ENC_SKIP || needs_extra_bit) {

                    memcpy(&buf[ret], temp, temp_size);

                    ret += temp_size;

                    temp_size = 0;

                } else

                    should_write_temp = 1;

            }



            if(needs_extra_bit) {

                flags = (mb->best_encoding == ENC_V4) << 31;

                bits = 1;

            }



            if(mb->best_encoding == ENC_V1)

                temp[temp_size++] = mb->v1_vector;

            else if(mb->best_encoding == ENC_V4)

                for(z = 0; z < 4; z++)

                    temp[temp_size++] = mb->v4_vector[v4][z];



            if(should_write_temp) {

                memcpy(&buf[ret], temp, temp_size);

                ret += temp_size;

                temp_size = 0;

            }

        }



        if(bits > 0) {

            AV_WB32(&buf[ret], flags);

            ret += 4;

            memcpy(&buf[ret], temp, temp_size);

            ret += temp_size;

        }



        write_chunk_header(buf + header_ofs, 0x31, ret - header_ofs - CHUNK_HEADER_SIZE);



        break;

    }



    return ret;

}
