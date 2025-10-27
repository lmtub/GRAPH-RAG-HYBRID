int av_vsrc_buffer_add_frame2(AVFilterContext *buffer_filter, AVFrame *frame,

                              int64_t pts, AVRational pixel_aspect, int width,

                              int height, enum PixelFormat  pix_fmt,

                              const char *sws_param)

{

    BufferSourceContext *c = buffer_filter->priv;

    int ret;



    if (c->has_frame) {

        av_log(buffer_filter, AV_LOG_ERROR,

               "Buffering several frames is not supported. "

               "Please consume all available frames before adding a new one.\n"

            );

        //return -1;

    }



    if(width != c->w || height != c->h || pix_fmt != c->pix_fmt){

        AVFilterContext *scale= buffer_filter->outputs[0]->dst;

        AVFilterLink *link;



        av_log(buffer_filter, AV_LOG_INFO, "Changing filter graph input to accept %dx%d %d (%d %d)\n",

               width,height,pix_fmt, c->pix_fmt, scale->outputs[0]->format);



        if(!scale || strcmp(scale->filter->name,"scale")){

            AVFilter *f= avfilter_get_by_name("scale");



            av_log(buffer_filter, AV_LOG_INFO, "Inserting scaler filter\n");

            if(avfilter_open(&scale, f, "Input equalizer") < 0)

                return -1;



            if((ret=avfilter_init_filter(scale, sws_param, NULL))<0){

                avfilter_free(scale);

                return ret;

            }



            if((ret=avfilter_insert_filter(buffer_filter->outputs[0], scale, 0, 0))<0){

                avfilter_free(scale);

                return ret;

            }



            scale->outputs[0]->format= c->pix_fmt;

        }



        c->pix_fmt= scale->inputs[0]->format= pix_fmt;

        c->w= scale->inputs[0]->w= width;

        c->h= scale->inputs[0]->h= height;



        link= scale->outputs[0];

        if ((ret =  link->srcpad->config_props(link)) < 0)

            return ret;

    }



    memcpy(c->frame.data    , frame->data    , sizeof(frame->data));

    memcpy(c->frame.linesize, frame->linesize, sizeof(frame->linesize));

    c->frame.interlaced_frame= frame->interlaced_frame;

    c->frame.top_field_first = frame->top_field_first;

    c->frame.key_frame = frame->key_frame;

    c->frame.pict_type = frame->pict_type;

    c->pts = pts;

    c->pixel_aspect = pixel_aspect;

    c->has_frame = 1;



    return 0;

}
