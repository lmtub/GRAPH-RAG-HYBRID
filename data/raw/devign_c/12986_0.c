static int ac3_eac3_probe(AVProbeData *p, enum CodecID expected_codec_id)

{

    int max_frames, first_frames = 0, frames;

    uint8_t *buf, *buf2, *end;

    AC3HeaderInfo hdr;

    GetBitContext gbc;

    enum CodecID codec_id = CODEC_ID_AC3;



    max_frames = 0;

    buf = p->buf;

    end = buf + p->buf_size;



    for(; buf < end; buf++) {

        buf2 = buf;



        for(frames = 0; buf2 < end; frames++) {

            init_get_bits(&gbc, buf2, 54);

            if(avpriv_ac3_parse_header(&gbc, &hdr) < 0)

                break;

            if(buf2 + hdr.frame_size > end ||

               av_crc(av_crc_get_table(AV_CRC_16_ANSI), 0, buf2 + 2, hdr.frame_size - 2))

                break;

            if (hdr.bitstream_id > 10)

                codec_id = CODEC_ID_EAC3;

            buf2 += hdr.frame_size;

        }

        max_frames = FFMAX(max_frames, frames);

        if(buf == p->buf)

            first_frames = frames;

    }

    if(codec_id != expected_codec_id) return 0;

    // keep this in sync with mp3 probe, both need to avoid

    // issues with MPEG-files!

    if   (first_frames>=4) return AVPROBE_SCORE_MAX/2+1;

    else if(max_frames>500)return AVPROBE_SCORE_MAX/2;

    else if(max_frames>=4) return AVPROBE_SCORE_MAX/4;

    else if(max_frames>=1) return 1;

    else                   return 0;

}
