int match_ext(const char *filename, const char *extensions)

{

    const char *ext, *p;

    char ext1[32], *q;



    if(!filename)

        return 0;

    

    ext = strrchr(filename, '.');

    if (ext) {

        ext++;

        p = extensions;

        for(;;) {

            q = ext1;

            while (*p != '\0' && *p != ',') 

                *q++ = *p++;

            *q = '\0';

            if (!strcasecmp(ext1, ext)) 

                return 1;

            if (*p == '\0') 

                break;

            p++;

        }

    }

    return 0;

}
