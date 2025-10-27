static char *qemu_rbd_parse_clientname(const char *conf, char *clientname)

{

    const char *p = conf;



    while (*p) {

        int len;

        const char *end = strchr(p, ':');



        if (end) {

            len = end - p;

        } else {

            len = strlen(p);

        }



        if (strncmp(p, "id=", 3) == 0) {

            len -= 3;

            strncpy(clientname, p + 3, len);

            clientname[len] = '\0';

            return clientname;

        }

        if (end == NULL) {

            break;

        }

        p = end + 1;

    }

    return NULL;

}
