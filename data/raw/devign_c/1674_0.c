static int img_set_parameters(AVFormatContext *s, AVFormatParameters *ap)

{

    VideoData *img = s->priv_data;

    AVStream *st;

    AVImageFormat *img_fmt;

    int i;



    /* find output image format */

    if (ap && ap->image_format) {

        img_fmt = ap->image_format;

    } else {

        img_fmt = guess_image_format(s->filename);

    }

    if (!img_fmt)

        return -1;



    if (s->nb_streams != 1)

        return -1;



    st = s->streams[0];

    /* we select the first matching format */

    for(i=0;i<PIX_FMT_NB;i++) {

        if (img_fmt->supported_pixel_formats & (1 << i))

            break;

    }

    if (i >= PIX_FMT_NB)

        return -1;

    img->img_fmt = img_fmt;

    img->pix_fmt = i;

    st->codec->pix_fmt = img->pix_fmt;

    return 0;

}
