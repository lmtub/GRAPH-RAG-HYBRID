static int foreach_device_config(int type, int (*func)(const char *cmdline))

{

    struct device_config *conf;

    int rc;



    TAILQ_FOREACH(conf, &device_configs, next) {

        if (conf->type != type)

            continue;

        rc = func(conf->cmdline);

        if (0 != rc)

            return rc;

    }

    return 0;

}
