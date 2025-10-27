int av_strerror(int errnum, char *errbuf, size_t errbuf_size)

{

    int ret = 0, i;

    struct error_entry *entry = NULL;



    for (i = 0; i < FF_ARRAY_ELEMS(error_entries); i++) {

        if (errnum == error_entries[i].num) {

            entry = &error_entries[i];

            break;

        }

    }

    if (entry) {

        av_strlcpy(errbuf, entry->str, errbuf_size);

    } else {

#if HAVE_STRERROR_R

        ret = strerror_r(AVUNERROR(errnum), errbuf, errbuf_size);

#else

        ret = -1;

#endif

        if (ret < 0)

            snprintf(errbuf, errbuf_size, "Error number %d occurred", errnum);

    }



    return ret;

}
