static int process_ipmovie_chunk(IPMVEContext *s, ByteIOContext *pb,

    AVPacket *pkt)

{

    unsigned char chunk_preamble[CHUNK_PREAMBLE_SIZE];

    int chunk_type;

    int chunk_size;

    unsigned char opcode_preamble[OPCODE_PREAMBLE_SIZE];

    unsigned char opcode_type;

    unsigned char opcode_version;

    int opcode_size;

    unsigned char scratch[1024];

    int i, j;

    int first_color, last_color;

    int audio_flags;

    unsigned char r, g, b;



    /* see if there are any pending packets */

    chunk_type = load_ipmovie_packet(s, pb, pkt);

    if ((chunk_type == CHUNK_VIDEO) && (chunk_type != CHUNK_DONE))

        return chunk_type;



    /* read the next chunk, wherever the file happens to be pointing */

    if (url_feof(pb))

        return CHUNK_EOF;

    if (get_buffer(pb, chunk_preamble, CHUNK_PREAMBLE_SIZE) !=

        CHUNK_PREAMBLE_SIZE)

        return CHUNK_BAD;

    chunk_size = AV_RL16(&chunk_preamble[0]);

    chunk_type = AV_RL16(&chunk_preamble[2]);



    debug_ipmovie("chunk type 0x%04X, 0x%04X bytes: ", chunk_type, chunk_size);



    switch (chunk_type) {



    case CHUNK_INIT_AUDIO:

        debug_ipmovie("initialize audio\n");

        break;



    case CHUNK_AUDIO_ONLY:

        debug_ipmovie("audio only\n");

        break;



    case CHUNK_INIT_VIDEO:

        debug_ipmovie("initialize video\n");

        break;



    case CHUNK_VIDEO:

        debug_ipmovie("video (and audio)\n");

        break;



    case CHUNK_SHUTDOWN:

        debug_ipmovie("shutdown\n");

        break;



    case CHUNK_END:

        debug_ipmovie("end\n");

        break;



    default:

        debug_ipmovie("invalid chunk\n");

        chunk_type = CHUNK_BAD;

        break;



    }



    while ((chunk_size > 0) && (chunk_type != CHUNK_BAD)) {



        /* read the next chunk, wherever the file happens to be pointing */

       if (url_feof(pb)) {

            chunk_type = CHUNK_EOF;

            break;

        }

        if (get_buffer(pb, opcode_preamble, CHUNK_PREAMBLE_SIZE) !=

            CHUNK_PREAMBLE_SIZE) {

            chunk_type = CHUNK_BAD;

            break;

        }



        opcode_size = AV_RL16(&opcode_preamble[0]);

        opcode_type = opcode_preamble[2];

        opcode_version = opcode_preamble[3];



        chunk_size -= OPCODE_PREAMBLE_SIZE;

        chunk_size -= opcode_size;

        if (chunk_size < 0) {

            debug_ipmovie("chunk_size countdown just went negative\n");

            chunk_type = CHUNK_BAD;

            break;

        }



        debug_ipmovie("  opcode type %02X, version %d, 0x%04X bytes: ",

            opcode_type, opcode_version, opcode_size);

        switch (opcode_type) {



        case OPCODE_END_OF_STREAM:

            debug_ipmovie("end of stream\n");

            url_fseek(pb, opcode_size, SEEK_CUR);

            break;



        case OPCODE_END_OF_CHUNK:

            debug_ipmovie("end of chunk\n");

            url_fseek(pb, opcode_size, SEEK_CUR);

            break;



        case OPCODE_CREATE_TIMER:

            debug_ipmovie("create timer\n");

            if ((opcode_version > 0) || (opcode_size > 6)) {

                debug_ipmovie("bad create_timer opcode\n");

                chunk_type = CHUNK_BAD;

                break;

            }

            if (get_buffer(pb, scratch, opcode_size) !=

                opcode_size) {

                chunk_type = CHUNK_BAD;

                break;

            }

            s->fps = 1000000.0 / (AV_RL32(&scratch[0]) * AV_RL16(&scratch[4]));

            s->frame_pts_inc = 90000 / s->fps;

            debug_ipmovie("  %.2f frames/second (timer div = %d, subdiv = %d)\n",

                s->fps, AV_RL32(&scratch[0]), AV_RL16(&scratch[4]));

            break;



        case OPCODE_INIT_AUDIO_BUFFERS:

            debug_ipmovie("initialize audio buffers\n");

            if ((opcode_version > 1) || (opcode_size > 10)) {

                debug_ipmovie("bad init_audio_buffers opcode\n");

                chunk_type = CHUNK_BAD;

                break;

            }

            if (get_buffer(pb, scratch, opcode_size) !=

                opcode_size) {

                chunk_type = CHUNK_BAD;

                break;

            }

            s->audio_sample_rate = AV_RL16(&scratch[4]);

            audio_flags = AV_RL16(&scratch[2]);

            /* bit 0 of the flags: 0 = mono, 1 = stereo */

            s->audio_channels = (audio_flags & 1) + 1;

            /* bit 1 of the flags: 0 = 8 bit, 1 = 16 bit */

            s->audio_bits = (((audio_flags >> 1) & 1) + 1) * 8;

            /* bit 2 indicates compressed audio in version 1 opcode */

            if ((opcode_version == 1) && (audio_flags & 0x4))

                s->audio_type = CODEC_ID_INTERPLAY_DPCM;

            else if (s->audio_bits == 16)

                s->audio_type = CODEC_ID_PCM_S16LE;

            else

                s->audio_type = CODEC_ID_PCM_U8;

            debug_ipmovie("audio: %d bits, %d Hz, %s, %s format\n",

                s->audio_bits,

                s->audio_sample_rate,

                (s->audio_channels == 2) ? "stereo" : "mono",

                (s->audio_type == CODEC_ID_INTERPLAY_DPCM) ?

                "Interplay audio" : "PCM");

            break;



        case OPCODE_START_STOP_AUDIO:

            debug_ipmovie("start/stop audio\n");

            url_fseek(pb, opcode_size, SEEK_CUR);

            break;



        case OPCODE_INIT_VIDEO_BUFFERS:

            debug_ipmovie("initialize video buffers\n");

            if ((opcode_version > 2) || (opcode_size > 8)) {

                debug_ipmovie("bad init_video_buffers opcode\n");

                chunk_type = CHUNK_BAD;

                break;

            }

            if (get_buffer(pb, scratch, opcode_size) !=

                opcode_size) {

                chunk_type = CHUNK_BAD;

                break;

            }

            s->video_width = AV_RL16(&scratch[0]) * 8;

            s->video_height = AV_RL16(&scratch[2]) * 8;

            debug_ipmovie("video resolution: %d x %d\n",

                s->video_width, s->video_height);

            break;



        case OPCODE_UNKNOWN_06:

        case OPCODE_UNKNOWN_0E:

        case OPCODE_UNKNOWN_10:

        case OPCODE_UNKNOWN_12:

        case OPCODE_UNKNOWN_13:

        case OPCODE_UNKNOWN_14:

        case OPCODE_UNKNOWN_15:

            debug_ipmovie("unknown (but documented) opcode %02X\n", opcode_type);

            url_fseek(pb, opcode_size, SEEK_CUR);

            break;



        case OPCODE_SEND_BUFFER:

            debug_ipmovie("send buffer\n");

            url_fseek(pb, opcode_size, SEEK_CUR);

            break;



        case OPCODE_AUDIO_FRAME:

            debug_ipmovie("audio frame\n");



            /* log position and move on for now */

            s->audio_chunk_offset = url_ftell(pb);

            s->audio_chunk_size = opcode_size;

            url_fseek(pb, opcode_size, SEEK_CUR);

            break;



        case OPCODE_SILENCE_FRAME:

            debug_ipmovie("silence frame\n");

            url_fseek(pb, opcode_size, SEEK_CUR);

            break;



        case OPCODE_INIT_VIDEO_MODE:

            debug_ipmovie("initialize video mode\n");

            url_fseek(pb, opcode_size, SEEK_CUR);

            break;



        case OPCODE_CREATE_GRADIENT:

            debug_ipmovie("create gradient\n");

            url_fseek(pb, opcode_size, SEEK_CUR);

            break;



        case OPCODE_SET_PALETTE:

            debug_ipmovie("set palette\n");

            /* check for the logical maximum palette size

             * (3 * 256 + 4 bytes) */

            if (opcode_size > 0x304) {

                debug_ipmovie("demux_ipmovie: set_palette opcode too large\n");

                chunk_type = CHUNK_BAD;

                break;

            }

            if (get_buffer(pb, scratch, opcode_size) != opcode_size) {

                chunk_type = CHUNK_BAD;

                break;

            }



            /* load the palette into internal data structure */

            first_color = AV_RL16(&scratch[0]);

            last_color = first_color + AV_RL16(&scratch[2]) - 1;

            /* sanity check (since they are 16 bit values) */

            if ((first_color > 0xFF) || (last_color > 0xFF)) {

                debug_ipmovie("demux_ipmovie: set_palette indices out of range (%d -> %d)\n",

                    first_color, last_color);

                chunk_type = CHUNK_BAD;

                break;

            }

            j = 4;  /* offset of first palette data */

            for (i = first_color; i <= last_color; i++) {

                /* the palette is stored as a 6-bit VGA palette, thus each

                 * component is shifted up to a 8-bit range */

                r = scratch[j++] * 4;

                g = scratch[j++] * 4;

                b = scratch[j++] * 4;

                s->palette_control.palette[i] = (r << 16) | (g << 8) | (b);

            }

            /* indicate a palette change */

            s->palette_control.palette_changed = 1;

            break;



        case OPCODE_SET_PALETTE_COMPRESSED:

            debug_ipmovie("set palette compressed\n");

            url_fseek(pb, opcode_size, SEEK_CUR);

            break;



        case OPCODE_SET_DECODING_MAP:

            debug_ipmovie("set decoding map\n");



            /* log position and move on for now */

            s->decode_map_chunk_offset = url_ftell(pb);

            s->decode_map_chunk_size = opcode_size;

            url_fseek(pb, opcode_size, SEEK_CUR);

            break;



        case OPCODE_VIDEO_DATA:

            debug_ipmovie("set video data\n");



            /* log position and move on for now */

            s->video_chunk_offset = url_ftell(pb);

            s->video_chunk_size = opcode_size;

            url_fseek(pb, opcode_size, SEEK_CUR);

            break;



        default:

            debug_ipmovie("*** unknown opcode type\n");

            chunk_type = CHUNK_BAD;

            break;



        }

    }



    /* make a note of where the stream is sitting */

    s->next_chunk_offset = url_ftell(pb);



    /* dispatch the first of any pending packets */

    if ((chunk_type == CHUNK_VIDEO) || (chunk_type == CHUNK_AUDIO_ONLY))

        chunk_type = load_ipmovie_packet(s, pb, pkt);



    return chunk_type;

}
