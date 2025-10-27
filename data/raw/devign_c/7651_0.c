void av_register_output_format(AVOutputFormat *format)

{

    AVOutputFormat **p = &first_oformat;



    while (*p != NULL)

        p = &(*p)->next;



    *p = format;

    format->next = NULL;

}
