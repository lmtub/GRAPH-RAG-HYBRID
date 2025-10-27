char *g_strdup_vprintf(const char *format, va_list ap)

{

    char ch, *s;

    size_t len;



    __coverity_string_null_sink__(format);

    __coverity_string_size_sink__(format);



    ch = *format;

    ch = *(char *)ap;



    s = __coverity_alloc_nosize__();

    __coverity_writeall__(s);

    __coverity_mark_as_afm_allocated__(s, AFM_free);



    return len;

}
