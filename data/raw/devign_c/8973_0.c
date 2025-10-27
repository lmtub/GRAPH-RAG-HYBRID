static void add_device_config(int type, const char *cmdline)

{

    struct device_config *conf;



    conf = qemu_mallocz(sizeof(*conf));

    conf->type = type;

    conf->cmdline = cmdline;

    TAILQ_INSERT_TAIL(&device_configs, conf, next);

}
