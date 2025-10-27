void avcodec_set_dimensions(AVCodecContext *s, int width, int height){

    s->coded_width = width;

    s->coded_height= height;

    s->width  = width;

    s->height = height;

}
