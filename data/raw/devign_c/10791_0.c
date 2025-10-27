int get_frame_filename(char *buf, int buf_size,

                       const char *path, int number)

{

    const char *p;

    char *q, buf1[20];

    int nd, len, c, percentd_found;



    q = buf;

    p = path;

    percentd_found = 0;

    for(;;) {

        c = *p++;

        if (c == '\0')

            break;

        if (c == '%') {

            do {

                nd = 0;

                while (isdigit(*p)) {

                    nd = nd * 10 + *p++ - '0';

                }

                c = *p++;

                if (c == '*' && nd > 0) {

                    // The nd field is actually the modulus

                    number = number % nd;

                    c = *p++;

                    nd = 0;

                }

            } while (isdigit(c));



            switch(c) {

            case '%':

                goto addchar;

            case 'd':

                if (percentd_found)

                    goto fail;

                percentd_found = 1;

                snprintf(buf1, sizeof(buf1), "%0*d", nd, number);

                len = strlen(buf1);

                if ((q - buf + len) > buf_size - 1)

                    goto fail;

                memcpy(q, buf1, len);

                q += len;

                break;

            default:

                goto fail;

            }

        } else {

        addchar:

            if ((q - buf) < buf_size - 1)

                *q++ = c;

        }

    }

    if (!percentd_found)

        goto fail;

    *q = '\0';

    return 0;

 fail:

    *q = '\0';

    return -1;

}
