static int replace_int_data_in_filename(char *buf, int buf_size, const char *filename, char placeholder, int64_t number)

{

    const char *p;

    char *q, buf1[20], c;

    int nd, len, addchar_count;

    int found_count = 0;



    q = buf;

    p = filename;

    for (;;) {

        c = *p;

        if (c == '\0')

            break;

        if (c == '%' && *(p+1) == '%')  // %%

            addchar_count = 2;

        else if (c == '%' && (av_isdigit(*(p+1)) || *(p+1) == placeholder)) {

            nd = 0;

            addchar_count = 1;

            while (av_isdigit(*(p + addchar_count))) {

                nd = nd * 10 + *(p + addchar_count) - '0';

                addchar_count++;

            }



            if (*(p + addchar_count) == placeholder) {

                len = snprintf(buf1, sizeof(buf1), "%0*"PRId64, (number < 0) ? nd : nd++, number);

                if (len < 1)  // returned error or empty buf1

                    goto fail;

                if ((q - buf + len) > buf_size - 1)

                    goto fail;

                memcpy(q, buf1, len);

                q += len;

                p += (addchar_count + 1);

                addchar_count = 0;

                found_count++;

            }



        } else

            addchar_count = 1;



        while (addchar_count--)

            if ((q - buf) < buf_size - 1)

                *q++ = *p++;

            else

                goto fail;

    }

    *q = '\0';

    return found_count;

fail:

    *q = '\0';

    return -1;

}
