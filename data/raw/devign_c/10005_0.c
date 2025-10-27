static int get_aac_sample_rates(AVFormatContext *s, AVCodecParameters *par,

                                int *sample_rate, int *output_sample_rate)

{

    MPEG4AudioConfig mp4ac;



    if (avpriv_mpeg4audio_get_config(&mp4ac, par->extradata,

                                     par->extradata_size * 8, 1) < 0) {

        av_log(s, AV_LOG_ERROR,

               "Error parsing AAC extradata, unable to determine samplerate.\n");

        return AVERROR(EINVAL);

    }



    *sample_rate        = mp4ac.sample_rate;

    *output_sample_rate = mp4ac.ext_sample_rate;

    return 0;

}
