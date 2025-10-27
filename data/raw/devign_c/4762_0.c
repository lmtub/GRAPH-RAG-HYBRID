static int mkv_write_codecprivate(AVFormatContext *s, AVIOContext *pb,

                                  AVCodecParameters *par,

                                  int native_id, int qt_id)

{

    AVIOContext *dyn_cp;

    uint8_t *codecpriv;

    int ret, codecpriv_size;



    ret = avio_open_dyn_buf(&dyn_cp);

    if (ret < 0)

        return ret;



    if (native_id) {

        ret = mkv_write_native_codecprivate(s, par, dyn_cp);

    } else if (par->codec_type == AVMEDIA_TYPE_VIDEO) {

        if (qt_id) {

            if (!par->codec_tag)

                par->codec_tag = ff_codec_get_tag(ff_codec_movvideo_tags,

                                                    par->codec_id);

            if (   ff_codec_get_id(ff_codec_movvideo_tags, par->codec_tag) == par->codec_id

                && (!par->extradata_size || ff_codec_get_id(ff_codec_movvideo_tags, AV_RL32(par->extradata + 4)) != par->codec_id)

            ) {

                int i;

                avio_wb32(dyn_cp, 0x5a + par->extradata_size);

                avio_wl32(dyn_cp, par->codec_tag);

                for(i = 0; i < 0x5a - 8; i++)

                    avio_w8(dyn_cp, 0);

            }

            avio_write(dyn_cp, par->extradata, par->extradata_size);

        } else {

            if (!ff_codec_get_tag(ff_codec_bmp_tags, par->codec_id))

                av_log(s, AV_LOG_WARNING, "codec %s is not supported by this format\n",

                       avcodec_get_name(par->codec_id));



            if (!par->codec_tag)

                par->codec_tag = ff_codec_get_tag(ff_codec_bmp_tags,

                                                  par->codec_id);

            if (!par->codec_tag && par->codec_id != AV_CODEC_ID_RAWVIDEO) {

                av_log(s, AV_LOG_ERROR, "No bmp codec tag found for codec %s\n",

                       avcodec_get_name(par->codec_id));

                ret = AVERROR(EINVAL);

            }



            ff_put_bmp_header(dyn_cp, par, ff_codec_bmp_tags, 0, 0);

        }

    } else if (par->codec_type == AVMEDIA_TYPE_AUDIO) {

        unsigned int tag;

        tag = ff_codec_get_tag(ff_codec_wav_tags, par->codec_id);

        if (!tag) {

            av_log(s, AV_LOG_ERROR, "No wav codec tag found for codec %s\n",

                   avcodec_get_name(par->codec_id));

            ret = AVERROR(EINVAL);

        }

        if (!par->codec_tag)

            par->codec_tag = tag;



        ff_put_wav_header(s, dyn_cp, par, FF_PUT_WAV_HEADER_FORCE_WAVEFORMATEX);

    }



    codecpriv_size = avio_close_dyn_buf(dyn_cp, &codecpriv);

    if (codecpriv_size)

        put_ebml_binary(pb, MATROSKA_ID_CODECPRIVATE, codecpriv,

                        codecpriv_size);

    av_free(codecpriv);

    return ret;

}
