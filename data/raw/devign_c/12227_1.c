static void get_sub_picture(CinepakEncContext *s, int x, int y, AVPicture *in, AVPicture *out)

{

    out->data[0] = in->data[0] + x + y * in->linesize[0];

    out->linesize[0] = in->linesize[0];



    if(s->pix_fmt == AV_PIX_FMT_YUV420P) {

        out->data[1] = in->data[1] + (x >> 1) + (y >> 1) * in->linesize[1];

        out->linesize[1] = in->linesize[1];



        out->data[2] = in->data[2] + (x >> 1) + (y >> 1) * in->linesize[2];

        out->linesize[2] = in->linesize[2];

    }

}
