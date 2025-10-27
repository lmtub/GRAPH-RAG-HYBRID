static void listflags(char *buf, int bufsize, uint32_t fbits,

    const char **featureset, uint32_t flags)

{

    const char **p = &featureset[31];

    char *q, *b, bit;

    int nc;



    b = 4 <= bufsize ? buf + (bufsize -= 3) - 1 : NULL;

    *buf = '\0';

    for (q = buf, bit = 31; fbits && bufsize; --p, fbits &= ~(1 << bit), --bit)

        if (fbits & 1 << bit && (*p || !flags)) {

            if (*p)

                nc = snprintf(q, bufsize, "%s%s", q == buf ? "" : " ", *p);

            else

                nc = snprintf(q, bufsize, "%s[%d]", q == buf ? "" : " ", bit);

            if (bufsize <= nc) {

                if (b)

                    sprintf(b, "...");

                return;

            }

            q += nc;

            bufsize -= nc;

        }

}
