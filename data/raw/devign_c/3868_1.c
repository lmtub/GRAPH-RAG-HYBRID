int av_samples_get_buffer_size(int *linesize, int nb_channels, int nb_samples,

                               enum AVSampleFormat sample_fmt, int align)

{

    int line_size;

    int sample_size = av_get_bytes_per_sample(sample_fmt);

    int planar      = av_sample_fmt_is_planar(sample_fmt);



    /* validate parameter ranges */

    if (!sample_size || nb_samples <= 0 || nb_channels <= 0)




    /* auto-select alignment if not specified */

    if (!align) {



        align = 1;

        nb_samples = FFALIGN(nb_samples, 32);

    }



    /* check for integer overflow */

    if (nb_channels > INT_MAX / align ||

        (int64_t)nb_channels * nb_samples > (INT_MAX - (align * nb_channels)) / sample_size)




    line_size = planar ? FFALIGN(nb_samples * sample_size,               align) :

                         FFALIGN(nb_samples * sample_size * nb_channels, align);

    if (linesize)

        *linesize = line_size;



    return planar ? line_size * nb_channels : line_size;

}