static char *get_human_readable_size(char *buf, int buf_size, int64_t size)

{

    static const char suffixes[NB_SUFFIXES] = "KMGT";

    int64_t base;

    int i;



    if (size <= 999) {

        snprintf(buf, buf_size, "%" PRId64, size);

    } else {

        base = 1024;

        for (i = 0; i < NB_SUFFIXES; i++) {

            if (size < (10 * base)) {

                snprintf(buf, buf_size, "%0.1f%c",

                         (double)size / base,

                         suffixes[i]);

                break;

            } else if (size < (1000 * base) || i == (NB_SUFFIXES - 1)) {

                snprintf(buf, buf_size, "%" PRId64 "%c",

                         ((size + (base >> 1)) / base),

                         suffixes[i]);

                break;

            }

            base = base * 1024;

        }

    }

    return buf;

}
