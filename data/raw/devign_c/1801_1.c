static int config_props(AVFilterLink *link)

{

    YADIFContext *yadif = link->src->priv;



    link->time_base.num = link->src->inputs[0]->time_base.num;

    link->time_base.den = link->src->inputs[0]->time_base.den * 2;

    link->w             = link->src->inputs[0]->w;

    link->h             = link->src->inputs[0]->h;



    if(yadif->mode&1)

        link->frame_rate = av_mul_q(link->src->inputs[0]->frame_rate, (AVRational){2,1});



    return 0;

}
