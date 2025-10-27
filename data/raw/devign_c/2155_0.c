static void mp3_write_xing(AVFormatContext *s)

{

    MP3Context       *mp3 = s->priv_data;

    AVCodecContext *codec = s->streams[mp3->audio_stream_idx]->codec;

    int32_t        header;

    MPADecodeHeader  mpah;

    int srate_idx, i, channels;

    int bitrate_idx;

    int best_bitrate_idx;

    int best_bitrate_error = INT_MAX;

    int xing_offset;

    int ver = 0;

    int lsf, bytes_needed;



    if (!s->pb->seekable || !mp3->write_xing)

        return;



    for (i = 0; i < FF_ARRAY_ELEMS(avpriv_mpa_freq_tab); i++) {

        const uint16_t base_freq = avpriv_mpa_freq_tab[i];



        if      (codec->sample_rate == base_freq)     ver = 0x3; // MPEG 1

        else if (codec->sample_rate == base_freq / 2) ver = 0x2; // MPEG 2

        else if (codec->sample_rate == base_freq / 4) ver = 0x0; // MPEG 2.5

        else continue;



        srate_idx = i;

        break;

    }

    if (i == FF_ARRAY_ELEMS(avpriv_mpa_freq_tab)) {

        av_log(s, AV_LOG_WARNING, "Unsupported sample rate, not writing Xing "

               "header.\n");

        return;

    }



    switch (codec->channels) {

    case 1:  channels = MPA_MONO;                                          break;

    case 2:  channels = MPA_STEREO;                                        break;

    default: av_log(s, AV_LOG_WARNING, "Unsupported number of channels, "

                    "not writing Xing header.\n");

             return;

    }



    /* dummy MPEG audio header */

    header  =  0xff                                  << 24; // sync

    header |= (0x7 << 5 | ver << 3 | 0x1 << 1 | 0x1) << 16; // sync/audio-version/layer 3/no crc*/

    header |= (srate_idx << 2) << 8;

    header |= channels << 6;



    lsf = !((header & (1 << 20) && header & (1 << 19)));



    xing_offset  = xing_offtbl[ver != 3][channels == 1];

    bytes_needed = 4              // header

                 + xing_offset

                 + 4              // xing tag

                 + 4              // frames/size/toc flags

                 + 4              // frames

                 + 4              // size

                 + XING_TOC_SIZE; // toc



    for (bitrate_idx = 1; bitrate_idx < 15; bitrate_idx++) {

        int bit_rate = 1000 * avpriv_mpa_bitrate_tab[lsf][3 - 1][bitrate_idx];

        int error    = FFABS(bit_rate - codec->bit_rate);



        if (error < best_bitrate_error){

            best_bitrate_error = error;

            best_bitrate_idx   = bitrate_idx;

        }

    }



    for (bitrate_idx = best_bitrate_idx; bitrate_idx < 15; bitrate_idx++) {

        int32_t mask = bitrate_idx << (4 + 8);

        header |= mask;



        avpriv_mpegaudio_decode_header(&mpah, header);



        if (bytes_needed <= mpah.frame_size)

            break;



        header &= ~mask;

    }



    avio_wb32(s->pb, header);



    avpriv_mpegaudio_decode_header(&mpah, header);



    av_assert0(mpah.frame_size >= XING_MAX_SIZE);



    ffio_fill(s->pb, 0, xing_offset);

    mp3->xing_offset = avio_tell(s->pb);

    ffio_wfourcc(s->pb, "Xing");

    avio_wb32(s->pb, 0x01 | 0x02 | 0x04);  // frames / size / TOC



    mp3->size = mpah.frame_size;

    mp3->want = 1;



    avio_wb32(s->pb, 0);  // frames

    avio_wb32(s->pb, 0);  // size



    // TOC

    for (i = 0; i < XING_TOC_SIZE; i++)

        avio_w8(s->pb, 255 * i / XING_TOC_SIZE);



    ffio_fill(s->pb, 0, mpah.frame_size - bytes_needed);

}
