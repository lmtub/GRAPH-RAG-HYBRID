static int get_real_id(const char *devpath, const char *idname, uint16_t *val)

{

    FILE *f;

    char name[128];

    long id;



    snprintf(name, sizeof(name), "%s%s", devpath, idname);

    f = fopen(name, "r");

    if (f == NULL) {

        error_report("%s: %s: %m", __func__, name);

        return -1;

    }

    if (fscanf(f, "%li\n", &id) == 1) {

        *val = id;

    } else {


        return -1;

    }




    return 0;

}