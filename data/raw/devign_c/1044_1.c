int av_dict_set(AVDictionary **pm, const char *key, const char *value,

                int flags)

{

    AVDictionary *m = *pm;

    AVDictionaryEntry *tag = av_dict_get(m, key, NULL, flags);

    char *oldval = NULL;



    if (!m)

        m = *pm = av_mallocz(sizeof(*m));



    if (tag) {

        if (flags & AV_DICT_DONT_OVERWRITE)

            return 0;

        if (flags & AV_DICT_APPEND)

            oldval = tag->value;

        else

            av_free(tag->value);

        av_free(tag->key);

        *tag = m->elems[--m->count];

    } else {

        AVDictionaryEntry *tmp = av_realloc(m->elems,

                                            (m->count + 1) * sizeof(*m->elems));

        if (tmp)

            m->elems = tmp;

        else

            return AVERROR(ENOMEM);

    }

    if (value) {

        if (flags & AV_DICT_DONT_STRDUP_KEY)

            m->elems[m->count].key = key;

        else

            m->elems[m->count].key = av_strdup(key);

        if (flags & AV_DICT_DONT_STRDUP_VAL) {

            m->elems[m->count].value = value;

        } else if (oldval && flags & AV_DICT_APPEND) {

            int len = strlen(oldval) + strlen(value) + 1;

            if (!(oldval = av_realloc(oldval, len)))

                return AVERROR(ENOMEM);

            av_strlcat(oldval, value, len);

            m->elems[m->count].value = oldval;

        } else

            m->elems[m->count].value = av_strdup(value);

        m->count++;

    }

    if (!m->count) {

        av_free(m->elems);

        av_freep(pm);

    }



    return 0;

}
