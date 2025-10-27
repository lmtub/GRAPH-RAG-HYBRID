int av_dict_set(AVDictionary **pm, const char *key, const char *value,

                int flags)

{

    AVDictionary *m = *pm;

    AVDictionaryEntry *tag = av_dict_get(m, key, NULL, flags);

    char *oldval = NULL, *copy_key = NULL, *copy_value = NULL;



    if (flags & AV_DICT_DONT_STRDUP_KEY)

        copy_key = (void *)key;

    else

        copy_key = av_strdup(key);

    if (flags & AV_DICT_DONT_STRDUP_VAL)

        copy_value = (void *)value;

    else if (copy_key)

        copy_value = av_strdup(value);

    if (!m)

        m = *pm = av_mallocz(sizeof(*m));

    if (!m || (key && !copy_key) || (value && !copy_value))

        goto err_out;



    if (tag) {

        if (flags & AV_DICT_DONT_OVERWRITE) {

            av_free(copy_key);

            av_free(copy_value);

            return 0;

        }

        if (flags & AV_DICT_APPEND)

            oldval = tag->value;

        else

            av_free(tag->value);

        av_free(tag->key);

        *tag = m->elems[--m->count];

    } else {

        AVDictionaryEntry *tmp = av_realloc(m->elems,

                                            (m->count + 1) * sizeof(*m->elems));

        if (!tmp)

            goto err_out;

        m->elems = tmp;

    }

    if (copy_value) {

        m->elems[m->count].key = copy_key;

        m->elems[m->count].value = copy_value;

        if (oldval && flags & AV_DICT_APPEND) {

            size_t len = strlen(oldval) + strlen(copy_value) + 1;

            char *newval = av_mallocz(len);

            if (!newval)

                goto err_out;

            av_strlcat(newval, oldval, len);

            av_freep(&oldval);

            av_strlcat(newval, copy_value, len);

            m->elems[m->count].value = newval;

            av_freep(&copy_value);

        }

        m->count++;

    } else {

        av_freep(&copy_key);

    }

    if (!m->count) {

        av_freep(&m->elems);

        av_freep(pm);

    }



    return 0;



err_out:

    if (m && !m->count) {

        av_freep(&m->elems);

        av_freep(pm);

    }

    av_free(copy_key);

    av_free(copy_value);

    return AVERROR(ENOMEM);

}
