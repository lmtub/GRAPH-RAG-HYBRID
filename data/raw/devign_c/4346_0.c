static const AVClass *filter_child_class_next(const AVClass *prev)

{

    AVFilter **f = NULL;



    while (prev && *(f = av_filter_next(f)))

        if ((*f)->priv_class == prev)

            break;



    while (*(f = av_filter_next(f)))

        if ((*f)->priv_class)

            return (*f)->priv_class;



    return NULL;

}
