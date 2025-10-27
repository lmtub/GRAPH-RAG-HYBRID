void av_register_input_format(AVInputFormat *format)

{

    AVInputFormat **p = last_iformat;



    format->next = NULL;

    while(*p || avpriv_atomic_ptr_cas((void * volatile *)p, NULL, format))

        p = &(*p)->next;

    last_iformat = &format->next;

}
