static int sdp_parse_fmtp_config_h264(AVStream * stream,

                                      PayloadContext * h264_data,

                                      char *attr, char *value)

{

    AVCodecContext *codec = stream->codec;

    assert(codec->codec_id == CODEC_ID_H264);

    assert(h264_data != NULL);



    if (!strcmp(attr, "packetization-mode")) {

        av_log(codec, AV_LOG_DEBUG, "RTP Packetization Mode: %d\n", atoi(value));

        h264_data->packetization_mode = atoi(value);

        /*

           Packetization Mode:

           0 or not present: Single NAL mode (Only nals from 1-23 are allowed)

           1: Non-interleaved Mode: 1-23, 24 (STAP-A), 28 (FU-A) are allowed.

           2: Interleaved Mode: 25 (STAP-B), 26 (MTAP16), 27 (MTAP24), 28 (FU-A), and 29 (FU-B) are allowed.

         */

        if (h264_data->packetization_mode > 1)

            av_log(codec, AV_LOG_ERROR,

                   "Interleaved RTP mode is not supported yet.");

    } else if (!strcmp(attr, "profile-level-id")) {

        if (strlen(value) == 6) {

            char buffer[3];

            // 6 characters=3 bytes, in hex.

            uint8_t profile_idc;

            uint8_t profile_iop;

            uint8_t level_idc;



            buffer[0] = value[0]; buffer[1] = value[1]; buffer[2] = '\0';

            profile_idc = strtol(buffer, NULL, 16);

            buffer[0] = value[2]; buffer[1] = value[3];

            profile_iop = strtol(buffer, NULL, 16);

            buffer[0] = value[4]; buffer[1] = value[5];

            level_idc = strtol(buffer, NULL, 16);



            // set the parameters...

            av_log(codec, AV_LOG_DEBUG,

                   "RTP Profile IDC: %x Profile IOP: %x Level: %x\n",

                   profile_idc, profile_iop, level_idc);

            h264_data->profile_idc = profile_idc;

            h264_data->profile_iop = profile_iop;

            h264_data->level_idc = level_idc;

        }

    } else  if (!strcmp(attr, "sprop-parameter-sets")) {

        uint8_t start_sequence[]= { 0, 0, 1 };

        codec->extradata_size= 0;

        codec->extradata= NULL;



        while (*value) {

            char base64packet[1024];

            uint8_t decoded_packet[1024];

            uint32_t packet_size;

            char *dst = base64packet;



            while (*value && *value != ','

                   && (dst - base64packet) < sizeof(base64packet) - 1) {

                *dst++ = *value++;

            }

            *dst++ = '\0';



            if (*value == ',')

                value++;



            packet_size= av_base64_decode(decoded_packet, base64packet, sizeof(decoded_packet));

            if (packet_size) {

                uint8_t *dest = av_malloc(packet_size + sizeof(start_sequence) +

                                         codec->extradata_size +

                                         FF_INPUT_BUFFER_PADDING_SIZE);

                if(dest)

                {

                    if(codec->extradata_size)

                    {

                        // av_realloc?

                        memcpy(dest, codec->extradata, codec->extradata_size);

                        av_free(codec->extradata);

                    }



                    memcpy(dest+codec->extradata_size, start_sequence, sizeof(start_sequence));

                    memcpy(dest+codec->extradata_size+sizeof(start_sequence), decoded_packet, packet_size);

                    memset(dest+codec->extradata_size+sizeof(start_sequence)+

                           packet_size, 0, FF_INPUT_BUFFER_PADDING_SIZE);



                    codec->extradata= dest;

                    codec->extradata_size+= sizeof(start_sequence)+packet_size;

                } else {

                    av_log(codec, AV_LOG_ERROR, "Unable to allocate memory for extradata!");

                    return AVERROR(ENOMEM);

                }

            }

        }

        av_log(codec, AV_LOG_DEBUG, "Extradata set to %p (size: %d)!", codec->extradata, codec->extradata_size);

    }

    return 0;

}
