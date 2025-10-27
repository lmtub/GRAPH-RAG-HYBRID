static int process_audio_header_elements(AVFormatContext *s)

{

    int inHeader = 1;

    EaDemuxContext *ea = s->priv_data;

    ByteIOContext *pb = &s->pb;

    int compression_type;



    ea->num_channels = 1;



    while (inHeader) {

        int inSubheader;

        uint8_t byte;

        byte = get_byte(pb);



        switch (byte) {

        case 0xFD:

            av_log (s, AV_LOG_INFO, "entered audio subheader\n");

            inSubheader = 1;

            while (inSubheader) {

                uint8_t subbyte;

                subbyte = get_byte(pb);



                switch (subbyte) {

                case 0x82:

                    ea->num_channels = read_arbitary(pb);

                    av_log (s, AV_LOG_INFO, "num_channels (element 0x82) set to 0x%08x\n", ea->num_channels);

                    break;

                case 0x83:

                    compression_type = read_arbitary(pb);

                    av_log (s, AV_LOG_INFO, "compression_type (element 0x83) set to 0x%08x\n", compression_type);

                    break;

                case 0x85:

                    ea->num_samples = read_arbitary(pb);

                    av_log (s, AV_LOG_INFO, "num_samples (element 0x85) set to 0x%08x\n", ea->num_samples);

                    break;

                case 0x8A:

                    av_log (s, AV_LOG_INFO, "element 0x%02x set to 0x%08x\n", subbyte, read_arbitary(pb));

                    av_log (s, AV_LOG_INFO, "exited audio subheader\n");

                    inSubheader = 0;

                    break;

                case 0xFF:

                    av_log (s, AV_LOG_INFO, "end of header block reached (within audio subheader)\n");

                    inSubheader = 0;

                    inHeader = 0;

                    break;

                default:

                    av_log (s, AV_LOG_INFO, "element 0x%02x set to 0x%08x\n", subbyte, read_arbitary(pb));

                    break;

                }

            }

            break;

        case 0xFF:

            av_log (s, AV_LOG_INFO, "end of header block reached\n");

            inHeader = 0;

            break;

        default:

            av_log (s, AV_LOG_INFO, "header element 0x%02x set to 0x%08x\n", byte, read_arbitary(pb));

            break;

        }

    }



    ea->audio_codec = CODEC_ID_ADPCM_EA;



    return 1;

}
